import pytest
from langchain_antfly.tools import create_graph_tools


class Graph:
    def query(self, query):
        return [{"query": query}]

    async def aquery_graph(self, query):
        return [{"query": query}]

    def traverse(self, keys, **kwargs):
        return [{"keys": keys, **kwargs}]


def test_graph_tools_are_bounded_read_interfaces():
    query, traverse = create_graph_tools(Graph())
    assert query.name == "antfly_graph_query"
    assert traverse.invoke({"keys": ["a"], "max_depth": 2}) == [{"keys": ["a"], "max_depth": 2, "limit": 20}]
    assert set(query.args) == {"query"}
    assert set(traverse.args) == {"keys", "max_depth", "limit"}


@pytest.mark.asyncio
async def test_async_graph_tools():
    query, traverse = create_graph_tools(Graph())
    assert await query.ainvoke({"query": '{"q":{}}'}) == [{"query": '{"q":{}}'}]
    assert (await traverse.ainvoke({"keys": ["a"]}))[0]["keys"] == ["a"]
