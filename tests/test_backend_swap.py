"""The comparison contract runs real built-in retrieval with deterministic vectors."""

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import InMemoryVectorStore

from examples.backend_swap import CORPUS, QUESTIONS, CachedEmbeddings, children, evaluate


class GoldenEmbeddings(Embeddings):
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [[float(text == d.page_content) for d in CORPUS] for text in texts]

    def embed_query(self, text):
        self.calls += 1
        return [float(expected == d.id) for d in CORPUS for q, expected in QUESTIONS if q == text]


def test_comparison_resolves_ids_sources_and_reuses_exact_vectors():
    provider = GoldenEmbeddings()
    cache = CachedEmbeddings(provider)
    parents = InMemoryStore()
    parents.mset([(d.id, d) for d in CORPUS])
    baseline = InMemoryVectorStore.from_documents(children(), cache)
    results = evaluate(baseline, parents)
    assert [r["parent_id"] for r in results] == [d.id for d in CORPUS]
    assert provider.calls == 6
    # A second backend consumes exactly the cached vectors with no provider available.
    attached = CachedEmbeddings(values=cache.values)
    second = InMemoryVectorStore.from_documents(children(), attached)
    assert evaluate(second, parents) == results
    parents.mdelete(["owner"])
    with pytest.raises(AssertionError, match="Parent or source metadata mismatch"):
        evaluate(second, parents)


def test_attach_cache_miss_does_not_silently_call_inference():
    with pytest.raises(ValueError, match="Cache miss"):
        CachedEmbeddings().embed_query("unknown")


def test_cleanup_uses_bearer_token_not_api_key_tuple(monkeypatch):
    import sys
    from types import SimpleNamespace

    from examples import backend_swap

    received = {}

    def client(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(drop_table=lambda name: None, close=lambda: None)

    monkeypatch.setenv("ANTFLY_TOKEN", "synthetic-test-token")
    monkeypatch.setattr(backend_swap, "AntflyClient", client)
    monkeypatch.setattr(sys, "argv", ["backend_swap", "cleanup", "--prefix", "backend_swap_auth"])
    backend_swap.main()
    assert received["token"] == "synthetic-test-token"
    assert "api_key" not in received
