from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
from langchain_antfly.vectorstores import AntflyVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests


class HTTPError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class MinimalEmbeddings(Embeddings):
    """Intentionally has no model, dimension, or provider metadata."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return {"foo": [1.0, 0.0], "bar": [0.0, 1.0], "baz": [-1.0, 0.0]}.get(text, [0.6, 0.8])


class MemoryClient:
    def __init__(self):
        self.tables = {}
        self.configs = {}
        self.calls = []
        self.indexes = SimpleNamespace(get=self.index_get, create=self.index_create)

    def create_table(self, table, indexes=None):
        if table in self.tables:
            raise HTTPError(409)
        self.tables[table] = {}
        self.configs[table] = copy.deepcopy(indexes or {})
        return {}

    def get_table(self, table):
        if table not in self.tables:
            raise HTTPError(404)
        return {}

    def index_get(self, table, index):
        if index not in self.configs.get(table, {}):
            raise HTTPError(404)
        return self.configs[table][index]

    def index_create(self, table, index, config):
        if index in self.configs[table]:
            raise HTTPError(409)
        self.configs[table][index] = copy.deepcopy(config)
        return config

    def batch(self, table, inserts=None, deletes=None, **kwargs):
        self.calls.append(("batch", kwargs))
        self.tables[table].update(copy.deepcopy(inserts or {}))
        for key in deletes or []:
            self.tables[table].pop(key, None)
        return {"status": "committed"}

    def get(self, table, key, **kwargs):
        if key not in self.tables[table]:
            raise HTTPError(404)
        return copy.deepcopy(self.tables[table][key])

    def query(self, table, **kwargs):
        self.calls.append(("query", kwargs))
        vector = next(iter(kwargs.get("embeddings", {}).values()), None)
        if vector is None:
            vector = MinimalEmbeddings().embed_query(kwargs.get("semantic_search", ""))
        index = (kwargs.get("indexes") or ["vectors"])[0]
        found = []
        for key, source in self.tables[table].items():
            if not matches({**source, "__key": key}, kwargs.get("filter_query")):
                continue
            target = source.get("_embeddings", {}).get(index) or MinimalEmbeddings().embed_query(source["text"])
            dot = sum(a * b for a, b in zip(vector, target, strict=True))
            distance = 1 - dot / (math.sqrt(sum(a * a for a in vector)) * math.sqrt(sum(a * a for a in target)))
            found.append({"_id": key, "_source": copy.deepcopy(source), "_distance": distance, "_score": -distance})
        found.sort(key=lambda hit: hit["_distance"])
        return {"responses": [{"hits": {"hits": found[: kwargs.get("limit", 10)]}}]}


def matches(source, query):
    if not query:
        return True
    if "ids" in query:
        return source.get("__key") in query["ids"]
    if "conjuncts" in query:
        return all(matches(source, part) for part in query["conjuncts"])
    if "term" in query:
        value = source
        for key in query["field"].split("."):
            value = value.get(key)
        return value == query["term"]
    raise AssertionError(query)


class TestStandardVectorStore(VectorStoreIntegrationTests):
    @property
    def has_sync(self):
        return True

    @property
    def has_async(self):
        return True

    @property
    def supports_get_by_ids(self):
        return True

    @pytest.fixture
    def vectorstore(self):
        client = MemoryClient()
        return AntflyVectorStore.from_texts([], self.get_embeddings(), client=client, table="standard", dimension=6)


def test_metadata_free_factory_ids_reuse_and_compatibility():
    client = MemoryClient()
    documents = [Document(id="supplied", page_content="foo", metadata={"x": [1]})]
    store = AntflyVectorStore.from_documents(documents, MinimalEmbeddings(), client=client, table="docs")
    assert store.similarity_search("foo")[0] == documents[0]
    assert documents[0].id == "supplied"
    reused = AntflyVectorStore.from_texts(["bar"], MinimalEmbeddings(), client=client, table="docs", ids=["second"])
    assert len(client.tables["docs"]) == 2
    assert reused.get_by_ids(["missing", "supplied", "supplied"])[0] == documents[0]
    with pytest.raises(ValueError, match="dimension"):
        AntflyVectorStore.from_texts(["x"], BadDimension(), client=client, table="docs")
    assert len(client.tables["docs"]) == 2


class BadDimension(MinimalEmbeddings):
    def embed_documents(self, texts):
        return [[1, 2, 3] for _ in texts]


def test_scores_threshold_mmr_and_fixed_filter():
    client = MemoryClient()
    store = AntflyVectorStore.from_texts(
        ["foo", "bar", "baz"],
        MinimalEmbeddings(),
        client=client,
        table="docs",
        ids=["a", "b", "c"],
        metadatas=[{"tenant": "a"}, {"tenant": "a"}, {"tenant": "b"}],
    )
    assert store.similarity_search_with_score("foo")[0][1] == 0
    assert [d.id for d, _ in store.similarity_search_with_relevance_scores("foo", score_threshold=0.9)] == ["a"]
    assert [d.id for d in store.max_marginal_relevance_search("foo", k=2, fetch_k=3, lambda_mult=0)] == ["a", "c"]
    store.filter = {"term": "a", "field": "metadata.tenant"}
    assert store.similarity_search("foo", filter={"term": "b", "field": "metadata.tenant"}) == []
    assert client.calls[-1][1]["filter_query"]["conjuncts"]


def test_reject_invalid_inputs_before_mutation():
    client = MemoryClient()
    store = AntflyVectorStore.create(client=client, table="docs", embedding=MinimalEmbeddings())
    with pytest.raises(ValueError, match="length"):
        store.add_texts(["foo"], ids=["a", "b"])
    with pytest.raises(ValueError, match="unique"):
        store.add_texts(["foo", "bar"], ids=["a", "a"])
    assert client.tables["docs"] == {}
    with pytest.raises(TypeError, match="Unsupported"):
        store.similarity_search("foo", typo=True)


@pytest.mark.asyncio
async def test_async_factory_retriever_and_delete():
    store = await AntflyVectorStore.afrom_documents(
        [Document(page_content="foo")], MinimalEmbeddings(), client=MemoryClient(), table="docs"
    )
    found = await store.as_retriever().ainvoke("foo")
    assert found[0].page_content == "foo"
    await store.adelete([found[0].id])
    assert await store.asimilarity_search("foo") == []


class TestLiveStandardVectorStore(VectorStoreIntegrationTests):
    """Run upstream contracts against a real server when explicitly configured."""

    @property
    def has_sync(self):
        return True

    @property
    def has_async(self):
        return True

    @property
    def supports_get_by_ids(self):
        return True

    @pytest.fixture
    def vectorstore(self):
        import os
        import uuid

        from antfly import AntflyClient

        url = os.environ.get("ANTFLY_TEST_URL")
        if not url:
            pytest.skip("Set ANTFLY_TEST_URL for real-server standard contracts")
        client = AntflyClient(url, token=os.getenv("ANTFLY_TEST_TOKEN"), timeout=120)
        table = "lc_standard_" + uuid.uuid4().hex[:12]
        try:
            yield AntflyVectorStore.from_texts([], self.get_embeddings(), client=client, table=table, dimension=6)
        finally:
            client.drop_table(table)


def test_live_managed_retrieval_mmr_and_update():
    import os
    import uuid

    from antfly import AntflyClient

    url, model = (
        os.environ.get("ANTFLY_TEST_URL"),
        (os.environ.get("ANTFLY_TEST_EMBEDDING_MODEL") or os.environ.get("ANTFLY_TEST_EMBED_MODEL")),
    )
    if not url or not model:
        pytest.skip("Set ANTFLY_TEST_URL and ANTFLY_TEST_EMBED_MODEL for managed-index validation")
    client = AntflyClient(url, token=os.getenv("ANTFLY_TEST_TOKEN"), timeout=120)
    table = "lc_managed_" + uuid.uuid4().hex[:12]
    try:
        store = AntflyVectorStore.from_texts(
            ["Antfly is a database.", "Bananas are yellow fruit.", "Databases store documents."],
            client=client,
            table=table,
            ids=["database", "fruit", "documents"],
            embedder={"provider": "antfly", "model": model},
        )
        assert store.similarity_search("database", k=1)[0].id in {"database", "documents"}
        assert len(store.max_marginal_relevance_search("database", k=2, fetch_k=3)) == 2
        store.add_texts(["A green cucumber vegetable."], ids=["fruit"])
        assert store.get_by_ids(["fruit"])[0].page_content == "A green cucumber vegetable."
        store.delete(["fruit"])
        assert "fruit" not in {doc.id for doc in store.similarity_search("fruit", k=3)}
    finally:
        client.drop_table(table)


def test_external_provenance_conflicts_and_no_model_metadata():
    client = MemoryClient()
    AntflyVectorStore.from_texts(
        ["foo"], MinimalEmbeddings(), client=client, table="docs", provenance={"model": "first"}
    )
    with pytest.raises(ValueError, match="provenance conflict"):
        AntflyVectorStore.from_texts(
            ["bar"], MinimalEmbeddings(), client=client, table="docs", provenance={"model": "second"}
        )
    assert len(client.tables["docs"]) == 1


def test_empty_factory_defers_dimension_without_probe():
    client = MemoryClient()
    store = AntflyVectorStore.from_texts([], MinimalEmbeddings(), client=client, table="docs")
    assert client.configs["docs"] == {}
    store.add_texts(["foo"])
    assert store.dimension == 2


def test_fixed_scope_applies_to_get_by_ids_and_no_write_replay():
    from langchain_antfly._utils import WriteOutcomeError

    client = MemoryClient()
    store = AntflyVectorStore.from_texts(
        ["foo", "bar"], MinimalEmbeddings(), client=client, table="docs", ids=["a", "b"]
    )
    store.filter = {"term": "foo", "field": "text"}
    assert [doc.id for doc in store.get_by_ids(["a", "b"])] == ["a"]
    calls = []

    def pending(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "committed_pending"}

    client.batch = pending
    with pytest.raises(WriteOutcomeError) as error:
        store.add_texts(["foo"])
    assert error.value.committed
    assert len(calls) == 1


def test_live_fixed_scope_and_metadata_filter():
    import os
    import uuid

    from antfly import AntflyClient
    from langchain_antfly.retrievers import AntflyRetriever

    url = os.environ.get("ANTFLY_TEST_URL")
    if not url:
        pytest.skip("Set ANTFLY_TEST_URL for real filter validation")
    client = AntflyClient(url, token=os.getenv("ANTFLY_TEST_TOKEN"), timeout=120)
    table = "lc_scope_" + uuid.uuid4().hex[:12]
    try:
        store = AntflyVectorStore.from_texts(
            ["foo", "bar", "baz"],
            MinimalEmbeddings(),
            client=client,
            table=table,
            ids=["a", "b", "c"],
            metadatas=[
                {"tenant": "red", "kind": "public"},
                {"tenant": "blue", "kind": "public"},
                {"tenant": "red", "kind": "private"},
            ],
        )
        store.filter = {"term": "red", "field": "metadata.tenant"}
        assert {doc.id for doc in store.similarity_search("foo")} == {"a", "c"}
        public = {"term": "public", "field": "metadata.kind"}
        assert [doc.id for doc in store.similarity_search("foo", filter=public)] == ["a"]
        assert {doc.id for doc in store.get_by_ids(["a", "b", "c"])} == {"a", "c"}
        assert [doc.id for doc in AntflyRetriever(vectorstore=store).invoke("foo", filter=public)] == ["a"]
    finally:
        client.drop_table(table)


def test_missing_or_reconfigured_resources_fail_before_query_provider_calls():
    class CountingEmbeddings(MinimalEmbeddings):
        def __init__(self):
            self.calls = 0

        def embed_query(self, text):
            self.calls += 1
            return super().embed_query(text)

    provider = CountingEmbeddings()
    client = MemoryClient()
    store = AntflyVectorStore.from_texts(["foo"], provider, client=client, table="docs", ids=["a"])
    before = provider.calls
    client.configs["docs"]["vectors"]["dimension"] = 3
    with pytest.raises(ValueError, match="dimension"):
        store.similarity_search("foo")
    assert provider.calls == before
    client.configs.pop("docs")
    client.tables.pop("docs")
    with pytest.raises(HTTPError) as error:
        store.similarity_search("foo")
    assert error.value.status_code == 404
    assert provider.calls == before
    with pytest.raises(HTTPError) as error:
        store.get_by_ids(["a"])
    assert error.value.status_code == 404


def test_empty_external_factory_can_learn_query_dimension_without_probe():
    client = MemoryClient()
    store = AntflyVectorStore.from_texts([], MinimalEmbeddings(), client=client, table="docs")
    assert store.similarity_search("foo") == []
    assert store.dimension == 2


def test_invalid_metadata_and_resource_fail_before_embedding_calls():
    client = MemoryClient()
    calls = []
    embedding = MinimalEmbeddings()
    embedding.embed_documents = lambda texts: calls.append(texts) or [[1.0, 0.0] for _ in texts]
    store = AntflyVectorStore.create(client=client, table="meta", embedding=embedding, dimension=2)
    with pytest.raises(ValueError):
        store.add_texts(["foo"], metadatas=[{"number": float("nan")}])
    assert calls == []
    client.configs["meta"]["vectors"]["external"] = False
    with pytest.raises(ValueError):
        store.add_texts(["foo"])
    assert calls == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_attached_relevance_uses_discovered_metric_on_first_call(asynchronous):
    client = MemoryClient()
    client.create_table(
        "t", indexes={"vectors": {"type": "embeddings", "field": "text", "distance_metric": "l2_squared"}}
    )
    client.query = lambda *a, **kw: {"hits": {"hits": [{"_id": "a", "_source": {"text": "a"}, "_distance": 0.5}]}}
    store = AntflyVectorStore(client=client, table="t")
    if asynchronous:
        result = await store.asimilarity_search_with_relevance_scores("q", score_threshold=0.7)
    else:
        result = store.similarity_search_with_relevance_scores("q", score_threshold=0.7)
    assert result == []


def test_managed_credentials_are_not_compared_to_redacted_configuration():
    client = MemoryClient()
    config = {"type": "embeddings", "field": "text", "embedder": {"provider": "openai", "model": "dense"}}
    client.create_table("t", indexes={"vectors": config})
    store = AntflyVectorStore(client=client, table="t", embedder={**config["embedder"], "api_key": "test-only"})
    store.validate()
    store.embedder["model"] = "other"
    with pytest.raises(ValueError, match="model"):
        store.validate()


@pytest.mark.parametrize("invalid", ["metadata", "index", "missing"])
def test_factory_rejects_invalid_target_or_metadata_before_inference(invalid):
    class Provider(MinimalEmbeddings):
        def embed_documents(self, texts):
            raise AssertionError("inference must not be called")

    client = MemoryClient()
    if invalid != "missing":
        client.create_table(
            "t", indexes={"vectors": {"type": "embeddings", "external": invalid != "index", "dimension": 2}}
        )
    metadata = [{"bad": object()}] if invalid == "metadata" else None
    with pytest.raises((ValueError, TypeError, HTTPError)):
        AntflyVectorStore.from_texts(
            ["q"], Provider(), metadatas=metadata, client=client, table="t", create_if_missing=False
        )


def test_managed_local_model_advertised_shape_is_validated():
    client = MemoryClient()
    client.create_table(
        "t",
        indexes={
            "vectors": {
                "type": "embeddings",
                "field": "text",
                "dimension": 3,
                "embedder": {"provider": "antfly", "model": "dense"},
            }
        },
    )
    client.models = lambda: {
        "embedders": {"dense": {"inputs": ["text"], "dense_output": {"minimum_dimensions": 1, "native_dimensions": 2}}}
    }
    with pytest.raises(ValueError, match="dense range"):
        AntflyVectorStore(client=client, table="t").validate()


def test_image_ingestion_rejects_invalid_metadata_before_provider():
    class Images(MinimalEmbeddings):
        def embed_images(self, *args, **kwargs):
            raise AssertionError("must not infer")

    store = AntflyVectorStore(client=MemoryClient(), table="missing", embedding=Images())
    with pytest.raises(TypeError):
        store.add_images([b"image"], metadatas=[{"bad": object()}])
    with pytest.raises(HTTPError):
        store.add_images([b"image"])
