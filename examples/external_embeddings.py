"""Use unchanged OpenAIEmbeddings against an explicit Antfly inference endpoint.

Optional install: uv sync --group interoperability
This example never selects OpenAI's cloud URL or downloads model weights.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import uuid
from pathlib import Path

import httpx
from langchain_antfly import AntflyVectorStore
from langchain_core.documents import Document

from ._client import client_from_env

DOCUMENTS = [
    Document(id="owner", page_content="Mira Chen owns operations and handles shipment escalation."),
    Document(id="shipment", page_content="The Q3 shipment target is 120 crates."),
    Document(id="warehouse", page_content="Apples are stored in the cool orchard warehouse."),
]
QUESTION = "Who owns operations?"


async def run(client, *, model: str, table_prefix: str) -> dict:
    # The optional provider is deliberately imported only by this explicit path.
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as error:
        raise RuntimeError("Install the optional provider with: uv sync --group interoperability") from error
    if not model:
        raise ValueError("Select an installed Antfly embedding model explicitly")
    base_url = client.base_url.rstrip("/") + "/ai/v1"
    tables = {mode: f"{table_prefix}_{mode}" for mode in ("sync", "async")}
    existing = {item.get("name", item.get("table_name")) for item in client.list_tables()}
    if existing.intersection(tables.values()):
        raise ValueError("Use a fresh dedicated prefix; this example never overwrites existing tables")
    attempted = []
    result = {}
    with httpx.Client(timeout=60) as sync_transport:
        async with httpx.AsyncClient(timeout=60) as async_transport:
            embedding = OpenAIEmbeddings(
                model=model,
                base_url=base_url,
                api_key=os.getenv("ANTFLY_TOKEN") or "antfly-local-unused",
                check_embedding_ctx_length=False,
                model_kwargs={"encoding_format": "float"},
                max_retries=0,
                http_client=sync_transport,
                http_async_client=async_transport,
            )
            try:
                attempted.append(tables["sync"])
                store = AntflyVectorStore.from_documents(
                    DOCUMENTS, embedding, client=client, table=tables["sync"], distance_metric="cosine"
                )
                sync_docs = store.as_retriever(search_kwargs={"k": 1}).invoke(QUESTION)
                attempted.append(tables["async"])
                async_store = await AntflyVectorStore.afrom_documents(
                    DOCUMENTS, embedding, client=client, table=tables["async"], distance_metric="cosine"
                )
                async_docs = await async_store.as_retriever(search_kwargs={"k": 1}).ainvoke(QUESTION)
                direct_async_vector = await embedding.aembed_query(QUESTION)
                if [doc.id for doc in sync_docs] != ["owner"] or [doc.id for doc in async_docs] != ["owner"]:
                    raise ValueError("Both third-party provider retrieval paths must rank the owner document first")
                result = {
                    "passed": True,
                    "provider": "langchain_openai.OpenAIEmbeddings",
                    "versions": {
                        name: importlib.metadata.version(name)
                        for name in ("langchain-openai", "openai", "langchain-core", "antfly-sdk")
                    },
                    "configuration": {
                        "model": model,
                        "base_url": base_url,
                        "check_embedding_ctx_length": False,
                        "encoding_format": "float",
                        "max_retries": 0,
                        "credential": "Antfly token or local dummy; omitted",
                    },
                    "sync_document_ids": [doc.id for doc in sync_docs],
                    "async_document_ids": [doc.id for doc in async_docs],
                    "direct_async_embedding_dimension": len(direct_async_vector),
                    "cloud_calls": 0,
                    "limitations": "OpenAIEmbeddings owns query/document task behavior; this proves unchanged provider compatibility, not identical task semantics to AntflyEmbeddings.",
                }
            finally:
                # Only the two absent-before-run, dedicated names are eligible.
                present = {item.get("name", item.get("table_name")) for item in client.list_tables()}
                removed = []
                for table in attempted:
                    if table in present:
                        client.drop_table(table)
                        removed.append(table)
                result["removed_owned_tables"] = removed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("ANTFLY_EMBEDDING_MODEL"))
    parser.add_argument("--prefix", default="external_embeddings_" + uuid.uuid4().hex[:12])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.model:
        parser.error("Set ANTFLY_EMBEDDING_MODEL or --model to an installed model")
    with client_from_env() as client:
        result = asyncio.run(run(client, model=args.model, table_prefix=args.prefix))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
