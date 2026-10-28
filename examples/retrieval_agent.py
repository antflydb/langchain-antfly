"""Run a retrieval tool inside an actual LangChain agent graph."""

import argparse
import asyncio
import os
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool


def build_agent(retriever: Any, model: Any) -> Any:
    @tool
    def search_reports(query: str) -> str:
        """Search the report corpus for evidence relevant to a question."""
        documents = retriever.invoke(query)
        return "\n\n".join(f"[{document.id}] {document.page_content}" for document in documents)

    return create_agent(
        model=model,
        tools=[search_reports],
        system_prompt="Search reports before answering. Cite evidence document IDs. Do not invent evidence.",
    )


def main() -> None:
    from langchain.chat_models import init_chat_model
    from langchain_antfly import AntflyEmbeddings, AntflyRetriever, AntflyVectorStore

    from ._client import client_from_env

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing LangChain provider:model with tool support")
    parser.add_argument("--table", default="antfly_quickstart")
    parser.add_argument("--async", dest="asynchronous", action="store_true")
    parser.add_argument("question")
    args = parser.parse_args()
    embedding_model = os.getenv("ANTFLY_EMBEDDING_MODEL")
    if not embedding_model:
        parser.error("Set ANTFLY_EMBEDDING_MODEL to the same model used to ingest this table")
    client = client_from_env()
    store = AntflyVectorStore(
        client=client, table=args.table, embedding=AntflyEmbeddings(client=client, model=embedding_model)
    )
    agent = build_agent(AntflyRetriever(vectorstore=store, mode="dense", k=3), init_chat_model(args.model))
    inputs = {"messages": [{"role": "user", "content": args.question}]}
    result = asyncio.run(agent.ainvoke(inputs)) if args.asynchronous else agent.invoke(inputs)
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
