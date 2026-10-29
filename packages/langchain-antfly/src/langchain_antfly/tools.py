"""Bounded read-oriented tools. Scope comes from the supplied store, never the model."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from ._utils import run_async


def create_graph_tools(graph_store: Any) -> list[StructuredTool]:
    """Create read-only native graph-query and traversal tools with fixed scope."""

    def graph_query(query: str) -> list[dict[str, Any]]:
        """Execute a JSON-encoded Antfly graph query in the fixed scope. Pass query as a string, not a mapping.

        Cypher is unsupported.
        """
        return graph_store.query(query)

    def graph_traverse(keys: list[str], max_depth: int = 1, limit: int = 20) -> list[dict[str, Any]]:
        """Find reachable nodes from encoded keys in the fixed namespace, with one selected path per destination."""
        return graph_store.traverse(keys, max_depth=max_depth, limit=limit)

    async def agraph_query(query: str) -> list[dict[str, Any]]:
        return await run_async(graph_store.query, query)

    async def agraph_traverse(keys: list[str], max_depth: int = 1, limit: int = 20) -> list[dict[str, Any]]:
        return await run_async(graph_store.traverse, keys, max_depth=max_depth, limit=limit)

    return [
        StructuredTool.from_function(graph_query, coroutine=agraph_query, name="antfly_graph_query"),
        StructuredTool.from_function(graph_traverse, coroutine=agraph_traverse, name="antfly_graph_traverse"),
    ]
