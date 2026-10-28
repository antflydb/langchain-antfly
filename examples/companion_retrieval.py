"""Hybrid retrieval, reranking and scoped graph context over persisted Orchard evidence.

No captioning/generation calls or model downloads. The input multimodal tables are
read-only; the dedicated graph table is the only resource this recipe creates.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

from langchain_antfly import AntflyGraphStore, AntflyReranker, AntflyRetriever
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

from ._client import client_from_env
from .multimodal.__main__ import attach
from .multimodal.evidence import Evidence, IncompleteEvidenceError, aresolve, resolve


def graph_document(parents: list[Document], children: list[Document], source_id: str) -> GraphDocument:
    """Link each source object to the other evidence objects in its report."""
    proxies = {c.metadata["doc_id"]: c.page_content for c in children}
    nodes = [
        Node(
            id=p.metadata["source_id"],
            type="Evidence",
            properties={
                "source_id": p.metadata["source_id"],
                "parent_id": p.id,
                "page_content": proxies[p.id],
                "source_sha256": p.metadata["source_sha256"],
            },
        )
        for p in parents
    ]
    return GraphDocument(
        nodes=nodes,
        relationships=[Relationship(source=a, target=b, type="REPORT_CONTEXT") for a in nodes for b in nodes if a != b],
        source=Document(page_content="Related evidence from the persisted Orchard report", metadata={"id": source_id}),
    )


def parent_candidates(documents: list[Document]) -> list[Document]:
    """Map graph parent identity and provenance without modifying caller documents."""
    mapped = []
    for document in documents:
        clone = document.model_copy(deep=True)
        reserved = clone.metadata.get("_antfly", {})
        properties = reserved.get("properties", {}) if isinstance(reserved, dict) else {}
        if isinstance(reserved, dict) and reserved.get("graph"):
            parent_id = properties.get("parent_id") if isinstance(properties, dict) else None
            if not isinstance(parent_id, str) or not parent_id:
                raise IncompleteEvidenceError(f"Graph candidate {document.id!r} has no nonempty properties.parent_id")
            if clone.metadata.get("doc_id", parent_id) != parent_id:
                raise IncompleteEvidenceError(f"Graph candidate {document.id!r} has conflicting parent references")
            clone.metadata["doc_id"] = parent_id
            for field in ("source_id", "source_sha256"):
                if field in properties:
                    value = properties[field]
                    if not isinstance(value, str) or not value or clone.metadata.get(field, value) != value:
                        raise IncompleteEvidenceError(
                            f"Graph candidate {document.id!r} has conflicting {field} provenance"
                        )
                    clone.metadata[field] = value
        mapped.append(clone)
    return mapped


def boundary_receipt(evidence: Evidence) -> dict:
    parts, images = evidence.content_parts()
    return {
        "parent_ids": [p.id for p in evidence.parents],
        "media": images,
        "image_parts": sum(part["type"] == "image_url" for part in parts),
    }


def run(client, *, prefix: str, model: str, rerank_model: str, graph_table: str) -> dict:
    started = time.monotonic()
    available = client.models()
    if rerank_model not in available.get("rerankers", {}):
        raise ValueError(
            f"Reranker {rerank_model!r} is not installed; available: {list(available.get('rerankers', {}))}"
        )
    vectors, parents, media = attach(client, prefix, model)
    manifest = parents.mget(["ingestion-manifest"])[0]
    if manifest is None:
        raise ValueError("Ingest the multimodal corpus first; completed ingestion manifest is missing")
    children = vectors.get_by_ids(manifest.metadata["children"])
    if len(children) != len(manifest.metadata["children"]):
        raise ValueError("Persisted children are missing; repair multimodal ingestion first")
    evidence = resolve(children, parents, media)
    tables = {t.get("name", t.get("table_name")) for t in client.list_tables()}
    if graph_table in tables:
        raise ValueError("Companion graph table already exists; use a new owned table or cleanup explicitly")
    # Preserve exact parent/child/media references from the primary example.
    graph = AntflyGraphStore.create(client, graph_table, namespace=prefix, max_depth=1, max_results=10)
    graph.add_graph_documents([graph_document(evidence.parents, children, prefix + ":companion")])
    graph.refresh_schema()
    reranker = AntflyReranker(client=client, model=rerank_model, top_n=3)
    hybrid = AntflyRetriever(vectorstore=vectors, mode="hybrid", k=3, fetch_k=3)
    ranked = AntflyRetriever(vectorstore=vectors, mode="hybrid", k=3, fetch_k=3, reranker=reranker)
    expansion_traces = []

    class TracedGraph:
        def expand_documents(self, documents, **kwargs):
            candidates = graph.expand_documents(documents, **kwargs)
            expansion_traces.append([d.model_dump() for d in candidates])
            return candidates

    expanded = AntflyRetriever(
        vectorstore=vectors,
        mode="hybrid",
        k=1,
        fetch_k=1,
        graph_store=TracedGraph(),
        graph_expansion={
            "seed_keys": lambda docs: [graph.node_key("Evidence", d.metadata["source_id"]) for d in docs],
            "max_depth": 1,
            "limit": 10,
            "edge_types": ["REPORT_CONTEXT"],
        },
        reranker=AntflyReranker(client=client, model=rerank_model, top_n=2),
    )
    question = "Who owns operations?"
    modes = {
        "full_text": AntflyRetriever(vectorstore=vectors, mode="text", k=3).invoke(question),
        "hybrid": hybrid.invoke(question),
        "hybrid_rerank": ranked.invoke(question),
        "standalone_rerank": reranker.rerank(question, children),
    }
    # Capture the actual seed-plus-neighbor candidates sent to reranking.
    modes["hybrid_graph_rerank"] = expanded.invoke(question)
    modes["async_hybrid_rerank"] = asyncio.run(ranked.ainvoke(question))
    modes["async_hybrid_graph_rerank"] = asyncio.run(expanded.ainvoke(question))
    for name, documents in modes.items():
        if not documents or documents[0].metadata.get("source_id") != "page-1":
            raise AssertionError(f"{name} did not return the expected operations owner evidence")
    for sync, asynchronous in (
        ("hybrid_rerank", "async_hybrid_rerank"),
        ("hybrid_graph_rerank", "async_hybrid_graph_rerank"),
    ):
        if [d.id for d in modes[sync]] != [d.id for d in modes[asynchronous]]:
            raise AssertionError(f"Sync/async {sync} disagree on evidence identity/order")
    if len(expansion_traces) != 2 or any(
        not candidates
        or candidates[0]["metadata"].get("source_id") != "page-1"
        or not any(d["metadata"].get("_antfly", {}).get("graph") for d in candidates)
        for candidates in expansion_traces
    ):
        raise AssertionError("Graph expansion did not preserve the seed and validated scoped neighbors")
    # Resolve the exact candidate sets captured above; never rerun top-k to guess
    # which graph references reached the multimodal evidence boundary.
    sync_candidates = parent_candidates([Document.model_validate(d) for d in expansion_traces[0]])
    async_candidates = parent_candidates([Document.model_validate(d) for d in expansion_traces[1]])
    boundary = {
        "sync": boundary_receipt(resolve(sync_candidates, parents, media)),
        "async": boundary_receipt(asyncio.run(aresolve(async_candidates, parents, media))),
    }
    if boundary["sync"] != boundary["async"]:
        raise AssertionError("Sync/async graph parent/media evidence disagrees")
    if set(boundary["sync"]["parent_ids"]) != set(manifest.metadata["parents"]):
        raise AssertionError("Graph candidates did not resolve the three expected persisted parents")
    if len(boundary["sync"]["parent_ids"]) != 3 or boundary["sync"]["image_parts"] != 2:
        raise AssertionError("Graph candidates must resolve three parents and two original image content parts")
    return {
        "passed": True,
        "command": [sys.executable, "-m", "examples.companion_retrieval", *sys.argv[1:]],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "models": {"embedding": model, "reranking": rerank_model},
        "model_capabilities": {
            "embedding": available.get("embedders", {}).get(model),
            "reranking": available["rerankers"][rerank_model],
        },
        "prefix": prefix,
        "owned_graph_table": graph_table,
        "question": question,
        "source_fingerprint": manifest.metadata["fingerprint"],
        "corpus_parent_ids": [p.id for p in evidence.parents],
        "retrieval": {name: [d.model_dump() for d in documents] for name, documents in modes.items()},
        "graph_schema": graph.get_structured_schema,
        "graph_expansion_candidates": expansion_traces,
        "graph_multimodal_boundary": boundary,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("langchain-antfly", "antfly-sdk", "langchain-core", "langchain-community")
        },
        "duration_seconds": time.monotonic() - started,
        "generation_calls": 0,
        "limitations": "Synthetic three-parent corpus; graph neighbors enrich seeds before reranking; final k=1 keeps best owner evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="orchard_demo")
    parser.add_argument("--model", default=os.getenv("ANTFLY_EMBEDDING_MODEL"))
    parser.add_argument("--rerank-model", default=os.getenv("ANTFLY_RERANK_MODEL"))
    parser.add_argument("--graph-table", default="orchard_companion_graph")
    parser.add_argument("--output", type=Path)
    parser.add_argument("command", choices=["run", "cleanup"])
    args = parser.parse_args()
    if args.graph_table in {f"{args.prefix}_{s}" for s in ("vectors", "parents", "media")}:
        parser.error("--graph-table must be independent of the input corpus tables")
    if args.command == "run" and (not args.model or not args.rerank_model):
        parser.error("Set embedding/reranking models explicitly through options or environment")
    with client_from_env() as client:
        if args.command == "cleanup":
            if args.graph_table in {t.get("name", t.get("table_name")) for t in client.list_tables()}:
                client.drop_table(args.graph_table)
            result = {"removed_owned_graph_table": args.graph_table}
        else:
            result = run(
                client,
                prefix=args.prefix,
                model=args.model,
                rerank_model=args.rerank_model,
                graph_table=args.graph_table,
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
