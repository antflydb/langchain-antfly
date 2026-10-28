import pytest
from langchain_antfly.retrievers import AntflyRetriever
from langchain_antfly.vectorstores import AntflyVectorStore
from test_vectorstores import MemoryClient, MinimalEmbeddings


def make_store():
    return AntflyVectorStore.from_texts(
        ["foo", "bar"], MinimalEmbeddings(), client=MemoryClient(), table="docs", ids=["a", "b"]
    )


def test_hybrid_query_and_scope_composition():
    store = make_store()
    retriever = AntflyRetriever(vectorstore=store, mode="hybrid", filter={"term": "foo", "field": "text"})
    documents = retriever.invoke("foo", filter={"term": "bar", "field": "text"})
    assert documents == []
    options = store.client.calls[-1][1]
    assert options["full_text_search"] == {"match": "foo", "field": "text"}
    assert options["embeddings"] == {"vectors": [1, 0]}
    assert options["merge_config"] == {"strategy": "rrf"}
    assert len(options["filter_query"]["conjuncts"]) == 2


@pytest.mark.asyncio
async def test_native_retriever_async_provenance():
    result = await AntflyRetriever(vectorstore=make_store(), mode="dense").ainvoke("foo")
    assert result[0].id == "a"
    assert result[0].metadata["_antfly"]["distance"] == 0


def test_constructor_rejects_ambiguous_configuration():
    with pytest.raises(ValueError, match="not both"):
        AntflyRetriever(vectorstore=make_store(), client=MemoryClient())


def test_graph_expansion_receives_effective_scope_and_cannot_replace_it():
    store = make_store()
    store.filter = {"term": "foo", "field": "text"}

    class Graph:
        def expand_documents(self, documents, **kwargs):
            assert kwargs["filter_query"] == {
                "conjuncts": [{"term": "foo", "field": "text"}, {"term": "foo", "field": "text"}]
            }
            raise ValueError("Cannot enforce scope")

    retriever = AntflyRetriever(
        vectorstore=store, graph_store=Graph(), graph_expansion={"limit": 4}, filter={"term": "foo", "field": "text"}
    )
    with pytest.raises(ValueError, match="scope"):
        retriever.invoke("foo")


def test_graph_scope_override_rejected_and_zero_k_does_no_work():
    store = make_store()
    with pytest.raises(ValueError, match="Unsupported graph"):
        AntflyRetriever(vectorstore=store, graph_store=object(), graph_expansion={"filter_query": {}})
    before = len(store.client.calls)
    assert AntflyRetriever(vectorstore=store, k=0).invoke("foo") == []
    assert len(store.client.calls) == before


def test_missing_index_is_validated_before_retriever_provider_call():
    class UnavailableProvider(MinimalEmbeddings):
        def embed_query(self, text):
            raise AssertionError("Provider must not run for an unavailable index")

    store = AntflyVectorStore(client=MemoryClient(), table="absent", embedding=UnavailableProvider())
    with pytest.raises(Exception) as error:
        AntflyRetriever(vectorstore=store).invoke("foo")
    assert getattr(error.value, "status_code", None) == 404
