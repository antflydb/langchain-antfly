"""Compare built-in in-memory retrieval with persistent Antfly using identical vectors."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from antfly import AntflyClient, AntflyNotFoundError
from langchain_antfly import AntflyDocumentStore, AntflyEmbeddings, AntflyVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import InMemoryVectorStore

CORPUS = [
    Document(
        id="owner",
        page_content="Mira Chen owns operations and handles shipment escalation.",
        metadata={"source": "orchard-report", "page": 1},
    ),
    Document(
        id="shipment",
        page_content="The Q3 shipment target is 120 crates.",
        metadata={"source": "orchard-report", "page": 2},
    ),
    Document(
        id="warehouse",
        page_content="Apples are stored in the cool orchard warehouse.",
        metadata={"source": "orchard-report", "page": 3},
    ),
]
QUESTIONS = [
    ("Who owns operations?", "owner"),
    ("What is the Q3 shipment target?", "shipment"),
    ("Where are apples stored?", "warehouse"),
]


class CachedEmbeddings(Embeddings):
    """Share exact text vectors; separate query/document cache keys preserve tasks."""

    def __init__(self, provider=None, values=None):
        self.provider, self.values = provider, dict(values or {})

    def _one(self, task, text):
        key = json.dumps([task, text])
        if key not in self.values:
            if self.provider is None:
                raise ValueError(f"Cache miss during attach: {task} {text!r}")
            self.values[key] = (
                self.provider.embed_query(text) if task == "query" else self.provider.embed_documents([text])[0]
            )
        return list(self.values[key])

    def embed_documents(self, texts):
        return [self._one("document", text) for text in texts]

    def embed_query(self, text):
        return self._one("query", text)


def children():
    return [Document(id=d.id, page_content=d.page_content, metadata={**d.metadata, "doc_id": d.id}) for d in CORPUS]


def evaluate(vectors, parents):
    """Exercise the same ordinary retriever and parent-store APIs for both backends."""
    result = []
    retriever = vectors.as_retriever(search_kwargs={"k": 1})
    for question, expected in QUESTIONS:
        hits = retriever.invoke(question)
        if len(hits) != 1 or hits[0].id != expected:
            raise AssertionError(f"Wrong child for {question!r}: {hits!r}")
        hit = hits[0]
        parent = parents.mget([hit.metadata["doc_id"]])[0]
        original = next(d for d in CORPUS if d.id == expected)
        if (
            parent is None
            or parent.id != expected
            or parent.page_content != original.page_content
            or parent.metadata != original.metadata
        ):
            raise AssertionError(f"Parent or source metadata mismatch for {expected}")
        if {k: hit.metadata[k] for k in original.metadata} != original.metadata:
            raise AssertionError("Child source metadata changed")
        result.append(
            {"question": question, "child_id": hit.id, "parent_id": parent.id, "source_metadata": parent.metadata}
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["compare", "attach", "cleanup"])
    parser.add_argument("--url", default=os.getenv("ANTFLY_URL", "http://localhost:8080"))
    parser.add_argument("--model", default=os.getenv("ANTFLY_EMBEDDING_MODEL"))
    parser.add_argument("--prefix", required=True, help="Unique owned prefix, beginning backend_swap_")
    parser.add_argument("--cache", type=Path, help="Exact embedding cache JSON; required except cleanup")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"backend_swap_[a-zA-Z0-9_]+", args.prefix):
        parser.error("Use a unique backend_swap_ prefix")
    client = AntflyClient(base_url=args.url, token=os.getenv("ANTFLY_TOKEN"))
    tables = [args.prefix + "_vectors", args.prefix + "_parents"]
    started = time.perf_counter()
    try:
        if args.phase == "cleanup":
            for table in tables:
                try:
                    client.drop_table(table)
                except AntflyNotFoundError:
                    pass
            return
        if args.cache is None:
            parser.error("--cache is required")
        if args.phase == "compare":
            if not args.model:
                parser.error("Supply --model or ANTFLY_EMBEDDING_MODEL")
            # Refuse to overwrite another run's resources.
            for table in tables:
                try:
                    client.get_table(table)
                except AntflyNotFoundError:
                    continue
                raise ValueError(f"Table already exists: {table}; use attach or another prefix")
            embedding = CachedEmbeddings(AntflyEmbeddings(client=client, model=args.model))
            baseline = InMemoryVectorStore.from_documents(children(), embedding)
            baseline_parents = InMemoryStore()
            baseline_parents.mset([(d.id, d) for d in CORPUS])
            baseline_result = evaluate(baseline, baseline_parents)
            vectors = AntflyVectorStore.from_documents(
                children(), embedding, client=client, table=tables[0], distance_metric="cosine"
            )
            parents = AntflyDocumentStore.create(client=client, table=tables[1])
            parents.mset([(d.id, d) for d in CORPUS])
        else:
            cached = json.loads(args.cache.read_text())
            embedding = CachedEmbeddings(values=cached["vectors"])
            baseline_result = None
            vectors = AntflyVectorStore(client=client, table=tables[0], embedding=embedding)
            parents = AntflyDocumentStore(client=client, table=tables[1])
        actual = evaluate(vectors, parents)
        if baseline_result is not None and actual != baseline_result:
            raise AssertionError("Backend results differ")
        if args.phase == "compare":
            args.cache.write_text(json.dumps({"model": args.model, "vectors": embedding.values}))
        report = {
            "phase": args.phase,
            "passed": True,
            "results": actual,
            "baseline_results": baseline_result,
            "vector_cache_entries": len(embedding.values),
            "retrieval_storage_external_services": {"in_memory": 0, "antfly": 1},
            "embedding_service": "Antfly inference shared by both during compare; no inference during attach",
            "restart": "fresh process attach; baseline state is lost without rebuilding",
            "duration_seconds": round(time.perf_counter() - started, 3),
        }
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
    finally:
        client.close()


if __name__ == "__main__":
    main()
