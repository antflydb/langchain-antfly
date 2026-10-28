"""Three documents, ordinary LangChain factory, one retrieval result."""

import argparse
import asyncio
import os

from langchain_antfly import AntflyEmbeddings, AntflyVectorStore
from langchain_core.documents import Document

from ._client import client_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.getenv("ANTFLY_EMBEDDING_MODEL"),
        help="Installed embedding model; defaults to ANTFLY_EMBEDDING_MODEL",
    )
    parser.add_argument("--table", default="antfly_quickstart")
    parser.add_argument("--async", dest="asynchronous", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    client = client_from_env()
    if args.cleanup:
        from antfly import AntflyNotFoundError

        try:
            client.drop_table(args.table)
        except AntflyNotFoundError:
            pass
        return
    if not args.model:
        parser.error("Set ANTFLY_EMBEDDING_MODEL or --model explicitly")
    documents = [
        Document(id="apples", page_content="Apples are stored in the cool orchard warehouse."),
        Document(id="owner", page_content="Mira Chen owns operations and handles shipment escalation."),
        Document(id="shipment", page_content="The Q3 shipment target is 120 crates."),
    ]
    embedding = AntflyEmbeddings(client=client, model=args.model)
    if args.asynchronous:

        async def run():
            store = await AntflyVectorStore.afrom_documents(
                documents, embedding, client=client, table=args.table, distance_metric="cosine"
            )
            return await store.as_retriever(search_kwargs={"k": 1}).ainvoke("Who owns operations?")

        results = asyncio.run(run())
    else:
        store = AntflyVectorStore.from_documents(
            documents, embedding, client=client, table=args.table, distance_metric="cosine"
        )
        results = store.as_retriever(search_kwargs={"k": 1}).invoke("Who owns operations?")
    if not results or results[0].id != "owner":
        raise RuntimeError(f"Expected the owner document, received {results!r}")
    print(f"[{results[0].id}] {results[0].page_content}")


if __name__ == "__main__":
    main()
