"""Evaluate three frozen graph labels over persisted Orchard evidence, without inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from langchain_antfly import AntflyByteStore, AntflyDocumentStore, AntflyGraphStore, AntflyVectorStore
from langchain_core.documents import Document

from ._client import client_from_env
from .companion_retrieval import graph_document
from .multimodal.corpus import FIXTURES, PDF
from .multimodal.evidence import resolve, sha256

QUESTIONS = FIXTURES / "graph-questions.json"
OWNER_KEY = "orchard-graph-evaluation-owner"


def evaluate(graph, manifest: dict) -> list[dict]:
    """Compare native operation outputs against labels frozen before execution."""
    records = []
    for question in manifest["questions"]:
        kind = question["operation"]
        if kind == "traverse":
            rows = graph.traverse([graph.node_key("Evidence", question["start"])], edge_types=["REPORT_CONTEXT"])
            actual = {"evidence": sorted(row["document"]["graph_id"] for row in rows)}
        elif kind == "count":
            rows = graph.query_graph(
                {
                    "count": {
                        "match": {
                            "anchor": "a",
                            "nodes": {"a": {}, "b": {}},
                            "edges": [{"from": "a", "to": "b", "types": ["REPORT_CONTEXT"]}],
                        },
                        "return": {"aggregates": {"relationships": {"count": "*"}}},
                    }
                }
            )
            if len(rows) != 1:
                raise ValueError("Expected one exact aggregate result")
            value = rows[0]["relationships"]
            actual = {"count": value["value"], "exact": value["exact"]}
        elif kind == "shortest_path":
            rows = graph.query_graph(
                {
                    "path": {
                        "shortest_path": {
                            "from": {"key": graph.node_key("Evidence", question["start"])},
                            "to": {"key": graph.node_key("Evidence", question["end"])},
                            "edge_types": ["REPORT_CONTEXT"],
                        }
                    }
                }
            )
            if len(rows) != 1:
                raise ValueError("Expected one shortest path")
            path = rows[0]["path"]
            identities = {graph.node_key("Evidence", source): source for source in ("page-1", "figure-1", "figure-2")}
            actual = {"evidence": [identities[node["key"]] for node in path["nodes"]], "length": path["length"]}
        else:
            raise ValueError(f"Unknown frozen graph operation: {kind}")
        records.append({**question, "actual": actual, "passed": actual == question["expected"], "rows": rows})
    return records


def run(client, *, prefix: str, graph_table: str) -> dict:
    started = time.monotonic()
    labels_bytes = QUESTIONS.read_bytes()
    labels = json.loads(labels_bytes)
    if graph_table in {f"{prefix}_{suffix}" for suffix in ("vectors", "parents", "media")}:
        raise ValueError("Graph evaluation table must be independent of corpus tables")
    tables = {item.get("name", item.get("table_name")) for item in client.list_tables()}
    if graph_table in tables:
        raise ValueError("Graph evaluation table already exists; choose a fresh table or cleanup its owned run")
    parents = AntflyDocumentStore(client=client, table=f"{prefix}_parents")
    media = AntflyByteStore(client=client, table=f"{prefix}_media")
    ingestion = parents.mget(["ingestion-manifest"])[0]
    if ingestion is None:
        raise ValueError("Completed persisted Orchard ingestion is required")
    children = AntflyVectorStore(client=client, table=f"{prefix}_vectors").get_by_ids(ingestion.metadata["children"])
    if {child.id for child in children} != set(ingestion.metadata["children"]):
        raise ValueError("Persisted corpus has missing children")
    evidence = resolve(children, parents, media)
    fixture_hash = sha256(PDF.read_bytes())
    if {p.metadata["source_id"] for p in evidence.parents} != {"page-1", "figure-1", "figure-2"} or any(
        p.metadata.get("source_sha256") != fixture_hash for p in evidence.parents
    ):
        raise ValueError("Persisted corpus does not match the frozen Orchard fixture")
    corpus = {
        "parents": [p.model_dump(mode="json") for p in sorted(evidence.parents, key=lambda p: p.id)],
        "children": [c.model_dump(mode="json") for c in sorted(children, key=lambda c: c.id)],
        "media": {key: sha256(value) for key, value in sorted(evidence.media.items())},
    }
    graph = AntflyGraphStore.create(client, graph_table, namespace=prefix, max_depth=2, max_results=20, num_shards=2)
    AntflyDocumentStore(client=client, table=graph_table).mset(
        [
            (
                OWNER_KEY,
                Document(
                    page_content="Owned supplementary graph evaluation", metadata={"role": OWNER_KEY, "prefix": prefix}
                ),
            )
        ]
    )
    graph.add_graph_documents([graph_document(evidence.parents, children, prefix + ":graph-evaluation")])
    records = evaluate(graph, labels)
    return {
        "passed": all(record["passed"] for record in records),
        "evaluation_revision": labels["evaluation_revision"],
        "questions_sha256": sha256(labels_bytes),
        "original_twenty_labels_sha256": sha256((FIXTURES / "answers.json").read_bytes()),
        "fixture_sha256": fixture_hash,
        "corpus_sha256": sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":")).encode()),
        "ingestion_manifest_sha256": sha256(json.dumps(ingestion.model_dump(mode="json"), sort_keys=True).encode()),
        "source_fingerprint": ingestion.metadata["fingerprint"],
        "owned_graph_table": graph_table,
        "prefix": prefix,
        "records": records,
        "model_calls": 0,
        "generation_calls": 0,
        "duration_seconds": time.monotonic() - started,
        "limitations": "Three synthetic graph contract cases supplement the unchanged twenty retrieval/answer labels; no answer generation.",
    }


def cleanup(client, *, prefix: str, graph_table: str) -> dict:
    if graph_table in {f"{prefix}_{suffix}" for suffix in ("vectors", "parents", "media")}:
        raise ValueError("Refusing to remove a corpus table")
    tables = {item.get("name", item.get("table_name")) for item in client.list_tables()}
    if graph_table in tables:
        owner = AntflyDocumentStore(client=client, table=graph_table).mget([OWNER_KEY])[0]
        if owner is None or owner.metadata != {"role": OWNER_KEY, "prefix": prefix}:
            raise ValueError("Graph table is not owned by this supplementary evaluation/prefix")
        client.drop_table(graph_table)
    return {"removed_owned_graph_table": graph_table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "cleanup"])
    parser.add_argument("--prefix", default="orchard_demo")
    parser.add_argument("--graph-table", default="orchard_graph_evaluation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with client_from_env() as client:
        result = (run if args.command == "run" else cleanup)(client, prefix=args.prefix, graph_table=args.graph_table)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
