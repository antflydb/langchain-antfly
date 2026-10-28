"""Actual upstream embedding/cache interoperability and durable namespace isolation."""

import os
import uuid
from unittest.mock import patch

import pytest
from antfly import AntflyClient
from langchain_antfly import AntflyByteStore, AntflyVectorStore
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from test_stores import MemoryClient
from test_vectorstores import MemoryClient as VectorClient


@pytest.fixture(params=["deterministic", "live"])
def cache_backend(request):
    if request.param == "live":
        url = os.getenv("ANTFLY_TEST_URL")
        if not url:
            pytest.skip("Set ANTFLY_TEST_URL for real-server embedding cache interoperability")
        client = AntflyClient(url, token=os.getenv("ANTFLY_TEST_TOKEN"))
        table = "audit_interface_fix_gate_cache_" + uuid.uuid4().hex[:12]
    else:
        client, table = MemoryClient(), "cache"
    client.create_table(table)
    try:
        yield client, table
    finally:
        if request.param == "live":
            client.drop_table(table)
            client.close()


class TaskEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 1.0]


def cached(provider, client, table):
    return CacheBackedEmbeddings.from_bytes_store(
        provider,
        AntflyByteStore(client=client, table=table, namespace="révision/user/1/documents"),
        namespace="model-revision",
        query_embedding_cache=AntflyByteStore(client=client, table=table, namespace="révision/user/1/queries"),
        key_encoder="sha256",
    )


def test_cache_separates_query_document_tasks_and_survives_reattachment(cache_backend):
    client, table = cache_backend
    first = cached(TaskEmbeddings(), client, table)
    assert first.embed_documents(["same", "same"]) == [[1, 0], [1, 0]]
    assert first.embed_query("same") == [0, 1]
    second = cached(TaskEmbeddings(), client, table)
    with (
        patch.object(TaskEmbeddings, "embed_documents", side_effect=AssertionError("unexpected inference")),
        patch.object(TaskEmbeddings, "embed_query", side_effect=AssertionError("unexpected inference")),
    ):
        assert second.embed_documents(["same"]) == [[1, 0]]
        assert second.embed_query("same") == [0, 1]
    neighbor = AntflyByteStore(client=client, table=table, namespace="révision/user/10/documents")
    assert list(neighbor.yield_keys()) == []
    assert (
        len(list(AntflyByteStore(client=client, table=table, namespace="révision/user/1/documents").yield_keys())) == 1
    )


@pytest.mark.asyncio
async def test_upstream_provider_and_cache_work_unchanged_sync_async(cache_backend):
    client, table = cache_backend
    provider = DeterministicFakeEmbedding(size=8)
    expected = provider.embed_documents(["alpha", "beta"])
    first = cached(provider, client, table)
    assert first.embed_documents(["alpha", "beta"]) == expected
    expected_query = provider.embed_query("question")
    assert await first.aembed_query("question") == expected_query
    second = cached(DeterministicFakeEmbedding(size=8), client, table)
    with (
        patch.object(DeterministicFakeEmbedding, "embed_documents", side_effect=AssertionError("unexpected inference")),
        patch.object(DeterministicFakeEmbedding, "embed_query", side_effect=AssertionError("unexpected inference")),
    ):
        assert await second.aembed_documents(["beta", "alpha"]) == list(reversed(expected))
        assert second.embed_query("question") == expected_query


def test_unmodified_upstream_embedding_provider_ingests_into_standard_factory():
    # Exercises the provider itself, independently of its cached composition.
    provider = DeterministicFakeEmbedding(size=8)
    client = VectorClient()
    store = AntflyVectorStore.from_texts(["alpha", "beta"], provider, client=client, table="upstream", ids=["a", "b"])
    assert store.similarity_search("alpha", k=1)[0].id == "a"


@pytest.mark.skipif(
    not os.getenv("ANTFLY_TEST_URL"), reason="Requires live Antfly for fresh-process persistent cache proof"
)
def test_cache_reopens_in_fresh_process_with_no_provider_available():
    import subprocess
    import sys

    client = AntflyClient(os.environ["ANTFLY_TEST_URL"], token=os.getenv("ANTFLY_TEST_TOKEN"))
    table = "audit_interface_fix_gate_cache_" + uuid.uuid4().hex[:12]
    client.create_table(table)
    try:
        assert cached(TaskEmbeddings(), client, table).embed_documents(["same"]) == [[1, 0]]
        code = """
import os, sys
from antfly import AntflyClient
from langchain_antfly import AntflyByteStore
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_core.embeddings import Embeddings
class Offline(Embeddings):
    def embed_documents(self, texts):
        raise AssertionError("a durable cache hit must not invoke inference")
    def embed_query(self, text):
        raise AssertionError("query must not be used for a document")
with AntflyClient(os.environ["ANTFLY_TEST_URL"], token=os.getenv("ANTFLY_TEST_TOKEN")) as client:
    cache = AntflyByteStore(client=client, table=sys.argv[1], namespace="révision/user/1/documents")
    embedding = CacheBackedEmbeddings.from_bytes_store(Offline(), cache, namespace="model-revision", key_encoder="sha256")
    assert embedding.embed_documents(["same"]) == [[1, 0]]
print("fresh-process persistent cache hit; no inference")
"""
        result = subprocess.run([sys.executable, "-c", code, table], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert "fresh-process persistent cache hit" in result.stdout
    finally:
        client.drop_table(table)
        client.close()
