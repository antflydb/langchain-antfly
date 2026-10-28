"""Native image vectors, independently of the summary-proxy example."""

import argparse
import json
import os

from langchain_antfly import AntflyByteStore, AntflyEmbeddings, AntflyVectorStore

from ._client import client_from_env
from .multimodal.corpus import extract
from .multimodal.evidence import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.getenv("ANTFLY_IMAGE_MODEL"),
        help="Installed image/text model; defaults to ANTFLY_IMAGE_MODEL",
    )
    parser.add_argument("--table", default="orchard_native")
    parser.add_argument("command", choices=["ingest", "verify", "cleanup"])
    args = parser.parse_args()
    client = client_from_env()
    tables = {item.get("name", item.get("table_name")) for item in client.list_tables()}
    if args.command == "cleanup":
        for table in [args.table, f"{args.table}_media"]:
            if table in tables:
                client.drop_table(table)
        return
    if not args.model:
        parser.error("Set ANTFLY_IMAGE_MODEL or --model explicitly")
    embedding = AntflyEmbeddings(client=client, model=args.model)
    vectors = (
        AntflyVectorStore(client=client, table=args.table, embedding=embedding)
        if args.table in tables
        else AntflyVectorStore.create(client=client, table=args.table, embedding=embedding, distance_metric="cosine")
    )
    media = (
        AntflyByteStore(client=client, table=f"{args.table}_media")
        if f"{args.table}_media" in tables
        else AntflyByteStore.create(client=client, table=f"{args.table}_media")
    )
    parents, _, values = extract()
    images = [p for p in parents if p.metadata.get("media_key")]
    if args.command == "ingest":
        media.mset(list(values.items()))
        ids = vectors.add_images(
            [values[p.metadata["media_key"]] for p in images],
            ids=[p.id for p in images],
            metadatas=[p.metadata for p in images],
        )
        print(json.dumps({"mode": "native image embeddings; no summary proxies", "ids": ids}, indent=2))
        return
    results = []
    failures = []
    for parent in images:
        key = parent.metadata["media_key"]
        stored = media.mget([key])[0]
        if stored is None or sha256(stored) != parent.metadata["media_sha256"]:
            raise ValueError(f"Native image bytes missing or corrupt: {key}")
        matches = vectors.similarity_search_by_image(stored, k=1)
        if not matches or matches[0].id != parent.id:
            failures.append(f"Image self-retrieval failed: {parent.id}")
        results.append({"query_image_sha256": sha256(stored), "retrieved_ids": [p.id for p in matches]})
    text = vectors.similarity_search("a bar chart of quarterly shipments", k=1)
    if not text or text[0].metadata["source_id"] != "figure-1":
        failures.append("Text-to-image chart retrieval failed")
    print(
        json.dumps(
            {
                "passed": not failures,
                "failures": failures,
                "image_queries": results,
                "text_query_ids": [p.id for p in text],
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
