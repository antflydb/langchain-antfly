"""Compare proxy and native retrieval on identical frozen questions, without generation.

No vision calls or caption regeneration. Proxy results use the persisted ingestion.
Native text and original images share one multimodal embedding space. The small
fixture is an acceptance test; recall@1/@2 expose ranking hidden by recall@5.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from langchain_antfly import AntflyEmbeddings, AntflyVectorStore

from .multimodal.__main__ import attach, client_from_env
from .multimodal.corpus import FIXTURES, PDF, extract
from .multimodal.evidence import sha256


def score(required: list[str], retrieved: list[str]) -> float | None:
    return len(set(required) & set(retrieved)) / len(set(required)) if required else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="orchard")
    parser.add_argument("--native-table", default="orchard_eval_native")
    parser.add_argument("--native-model", default=os.getenv("ANTFLY_IMAGE_MODEL"))
    parser.add_argument("--proxy-model", default=os.getenv("ANTFLY_EMBEDDING_MODEL"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", choices=["ingest", "evaluate", "cleanup"])
    args = parser.parse_args()
    if not args.prefix.replace("_", "").isalnum():
        parser.error("--prefix must contain only letters, digits and underscores")
    if args.native_table in {f"{args.prefix}_{suffix}" for suffix in ("vectors", "parents", "media")}:
        parser.error("--native-table must be independent of all proxy tables")
    if args.command != "cleanup" and (not args.native_model or not args.proxy_model):
        parser.error("Set explicit native and proxy models through options or environment")
    with client_from_env() as client:
        tables = {item.get("name", item.get("table_name")) for item in client.list_tables()}
        if args.command == "cleanup":
            if args.native_table in tables:
                client.drop_table(args.native_table)
            return
        embedding = AntflyEmbeddings(client=client, model=args.native_model)
        if args.command == "ingest":
            parents, _, media = extract()
            if args.native_table in tables:
                raise ValueError(
                    "Native evaluation table already exists; choose a fresh dedicated table or clean it up"
                )
            native = AntflyVectorStore.create(
                client=client,
                table=args.native_table,
                embedding=embedding,
                distance_metric="cosine",
                provenance={"model": args.native_model},
            )
            text = [p for p in parents if not p.metadata.get("media_key")]
            images = [p for p in parents if p.metadata.get("media_key")]
            native.add_documents(text)
            native.add_images(
                [media[p.metadata["media_key"]] for p in images],
                ids=[p.id for p in images],
                metadatas=[p.metadata for p in images],
            )
            print(json.dumps({"ingested": len(parents), "native_model": args.native_model}))
            return
        native = AntflyVectorStore(
            client=client, table=args.native_table, embedding=embedding, provenance={"model": args.native_model}
        )
        proxy, parents_store, _ = attach(client, args.prefix, args.proxy_model)
        manifest = json.loads((FIXTURES / "answers.json").read_text())
        records = []
        for question in manifest["questions"]:
            row = {
                "id": question["id"],
                "category": question["category"],
                "question": question["question"],
                "required": question["parents"],
                "modes": {},
            }
            for mode, store in [("proxy", proxy), ("native", native)]:
                start = time.monotonic()
                documents = store.similarity_search(question["question"], k=5)
                if mode == "proxy":
                    keys = list(dict.fromkeys(d.metadata["doc_id"] for d in documents))
                    parents = parents_store.mget(keys)
                    if any(p is None for p in parents):
                        raise ValueError("Broken proxy parent reference; repair ingestion before evaluation")
                    retrieved = [p.metadata["source_id"] for p in parents]
                else:
                    retrieved = list(dict.fromkeys(d.metadata["source_id"] for d in documents))
                row["modes"][mode] = {
                    "retrieved": retrieved,
                    "latency_seconds": time.monotonic() - start,
                    "recall": {str(k): score(question["parents"], retrieved[:k]) for k in [1, 2, 5]},
                }
            records.append(row)
        metrics = {}
        for mode in ["proxy", "native"]:
            metrics[mode] = {}
            for category in ["all", *sorted({r["category"] for r in records})]:
                selected = [r for r in records if r["required"] and (category == "all" or r["category"] == category)]
                metrics[mode][category] = {
                    "answerable_questions": len(selected),
                    "mean_required_evidence_recall": {
                        str(k): sum(r["modes"][mode]["recall"][str(k)] for r in selected) / len(selected)
                        if selected
                        else None
                        for k in [1, 2, 5]
                    },
                }
        report = {
            "evaluation_revision": manifest["evaluation_revision"],
            "corpus_sha256": sha256(PDF.read_bytes()),
            "labels_sha256": sha256((FIXTURES / "answers.json").read_bytes()),
            "models": {"native": args.native_model, "proxy": args.proxy_model},
            "metrics": metrics,
            "questions": records,
            "limitations": [
                "Three parents: recall@5 is saturated; inspect recall@1 and recall@2.",
                "Negative questions have no evidence labels and are excluded from recall.",
                "Retrieval only: answer grounding and abstention require the generation evaluation.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
