"""Execute actual agent graphs using deterministic tool-calling test models."""

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from examples.agent_memory import UserContext, build_agent


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def model_call(name, args):
    return ToolModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1", "type": "tool_call"}]),
            AIMessage(content="Done"),
        ]
    )


def test_actual_agent_runtime_uses_trusted_user_context():
    store = InMemoryStore()
    agent = build_agent(store, model_call("remember_preference", {"key": "style", "value": "concise"}))
    agent.invoke({"messages": [{"role": "user", "content": "Remember concise style"}]}, context=UserContext("alice"))
    assert store.get(("users", "alice"), "style").value == {"value": "concise"}
    assert store.get(("users", "bob"), "style") is None
    recall = build_agent(store, model_call("recall_preference", {"key": "style"}))
    output = recall.invoke({"messages": [{"role": "user", "content": "Recall style"}]}, context=UserContext("bob"))
    assert output["messages"][-2].content == '{"found": false}'


async def test_actual_async_agent_runtime():
    store = InMemoryStore()
    agent = build_agent(store, model_call("remember_preference", {"key": "style", "value": "detailed"}))
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Remember detailed style"}]}, context=UserContext("bob")
    )
    assert store.get(("users", "bob"), "style").value == {"value": "detailed"}


def test_retrieval_tool_executes_in_agent_graph():
    from langchain_core.documents import Document

    from examples.retrieval_agent import build_agent as retrieval_agent

    class Retriever:
        def invoke(self, query):
            assert query == "operations owner"
            return [Document(id="page-1", page_content="Mira Chen owns operations")]

    agent = retrieval_agent(Retriever(), model_call("search_reports", {"query": "operations owner"}))
    result = agent.invoke({"messages": [{"role": "user", "content": "Who owns operations?"}]})
    assert "[page-1] Mira Chen" in result["messages"][-2].content
