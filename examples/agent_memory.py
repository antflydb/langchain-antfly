"""Two-user agent memory through ToolRuntime.store, using an existing chat provider."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool


@dataclass(frozen=True)
class UserContext:
    """Set by the trusted application, never chosen by model tool arguments."""

    user_id: str


@tool
def remember_preference(key: str, value: str, runtime: ToolRuntime[UserContext]) -> str:
    """Remember one preference for the currently authenticated user."""
    runtime.store.put(("users", runtime.context.user_id), key, {"value": value})
    return "Saved preference"


@tool
def recall_preference(key: str, runtime: ToolRuntime[UserContext]) -> str:
    """Recall a preference for the currently authenticated user."""
    item = runtime.store.get(("users", runtime.context.user_id), key)
    return json.dumps(item.value if item else {"found": False})


def build_agent(store: Any, model: Any) -> Any:
    return create_agent(
        model=model,
        tools=[remember_preference, recall_preference],
        store=store,
        context_schema=UserContext,
        system_prompt=(
            "Use the preference tools to save or recall preferences. Keys are literal names such as style or language. "
            "Use the key named by the user. After one save or recall call, report its result directly. "
            "Do not repeat the call or guess alternative keys. Never invent a saved preference."
        ),
    )


def main() -> None:
    from antfly import AntflyClient
    from langchain.chat_models import init_chat_model
    from langgraph_store_antfly import AntflyStore

    from ._client import client_from_env

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Existing LangChain provider:model; install its integration separately"
    )
    parser.add_argument("--table", default="agent_memory_demo")
    parser.add_argument("--user", required=True, help="Trusted application identity for this demo")
    parser.add_argument("--async", dest="asynchronous", action="store_true")
    parser.add_argument("message")
    args = parser.parse_args()
    client: AntflyClient = client_from_env()
    existing = {item.get("name", item.get("table_name")) for item in client.list_tables()}
    store = (
        AntflyStore(client=client, table=args.table)
        if args.table in existing
        else AntflyStore.create(client=client, table=args.table)
    )
    agent = build_agent(store, init_chat_model(args.model))
    inputs = {"messages": [{"role": "user", "content": args.message}]}
    context = UserContext(args.user)
    response = (
        asyncio.run(agent.ainvoke(inputs, context=context))
        if args.asynchronous
        else agent.invoke(inputs, context=context)
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
