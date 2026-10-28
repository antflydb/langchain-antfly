"""Persisted-corpus composition and scoped graph mapping acceptance."""

import json
import os
import sys
import uuid

import pytest
from antfly import AntflyClient
from langchain_core.documents import Document

from examples import companion_retrieval as recipe


def test_graph_context_uses_exact_persisted_parent_and_proxy_identity():
    parents = [
        Document(id="parent-a", page_content="original a", metadata={"source_id": "a", "source_sha256": "hash"}),
        Document(id="parent-b", page_content="original b", metadata={"source_id": "b", "source_sha256": "hash"}),
    ]
    children = [
        Document(id="child-a", page_content="proxy a", metadata={"doc_id": "parent-a"}),
        Document(id="child-b", page_content="proxy b", metadata={"doc_id": "parent-b"}),
    ]
    graph = recipe.graph_document(parents, children, "owned-source")
    assert graph.source.metadata["id"] == "owned-source"
    assert {(n.id, n.properties["parent_id"], n.properties["page_content"]) for n in graph.nodes} == {
        ("a", "parent-a", "proxy a"),
        ("b", "parent-b", "proxy b"),
    }
    assert {(e.source.id, e.target.id) for e in graph.relationships} == {("a", "b"), ("b", "a")}
    assert parents[0].page_content == "original a"


def test_graph_cleanup_cannot_alias_input_corpus(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["companion", "--graph-table", "orchard_demo_parents", "cleanup"])
    monkeypatch.setattr(recipe, "client_from_env", lambda: pytest.fail("must fail before connecting"))
    with pytest.raises(SystemExit, match="2"):
        recipe.main()


def test_missing_ingestion_fails_before_graph_resource_creation(monkeypatch):
    class Parents:
        def mget(self, keys):
            return [None]

    class Client:
        def models(self):
            return {"rerankers": {"rank": {}}}

        def list_tables(self):
            pytest.fail("graph provisioning must not start")

    monkeypatch.setattr(recipe, "attach", lambda *args: (None, Parents(), None))
    with pytest.raises(ValueError, match="manifest"):
        recipe.run(Client(), prefix="missing", model="embed", rerank_model="rank", graph_table="owned")


def test_live_persisted_corpus_companion(tmp_path):
    url = os.getenv("ANTFLY_TEST_URL")
    prefix = os.getenv("ANTFLY_TEST_CORPUS_PREFIX")
    model = os.getenv("ANTFLY_TEST_EMBEDDING_MODEL")
    rank = os.getenv("ANTFLY_TEST_RERANK_MODEL")
    if not all([url, model, rank]):
        pytest.skip("Requires installed-model live profile")
    owns_corpus = prefix is None
    prefix = prefix or "lc_companion_corpus_" + uuid.uuid4().hex[:12]
    table = "lc_companion_" + uuid.uuid4().hex[:12]
    with AntflyClient(url, token=os.getenv("ANTFLY_TEST_TOKEN"), timeout=120) as client:
        try:
            if owns_corpus:
                from examples.multimodal.__main__ import ingest

                ingest(client, prefix, model)
            result = recipe.run(client, prefix=prefix, model=model, rerank_model=rank, graph_table=table)
            assert result["passed"] and result["generation_calls"] == 0
            assert result["retrieval"]["full_text"]
            assert result["retrieval"]["async_hybrid_graph_rerank"]
            (tmp_path / "companion.json").write_text(json.dumps(result))
        finally:
            names = {t.get("name", t.get("table_name")) for t in client.list_tables()}
            if table in names:
                client.drop_table(table)
            if owns_corpus:
                for suffix in ("vectors", "parents", "media"):
                    if f"{prefix}_{suffix}" in names:
                        client.drop_table(f"{prefix}_{suffix}")


def graph_candidates_and_stores():
    from examples.multimodal.corpus import extract

    parents, children, media = extract()
    candidates = [children[0]] + [
        Document(
            id="graph:" + p.id,
            page_content="graph proxy",
            metadata={
                "_antfly": {
                    "graph": {"operation": "neighbors"},
                    "properties": {
                        "parent_id": p.id,
                        "source_id": p.metadata["source_id"],
                        "source_sha256": p.metadata["source_sha256"],
                    },
                },
            },
        )
        for p in parents[1:]
    ]

    class Store:
        def __init__(self, values):
            self.values = values

        def mget(self, keys):
            return [self.values.get(k) for k in keys]

        async def amget(self, keys):
            return self.mget(keys)

    return candidates, Store({p.id: p for p in parents}), Store(media)


async def test_graph_candidates_resolve_original_media_with_sync_async_parity_without_mutation():
    from examples.multimodal.evidence import aresolve, resolve

    candidates, parents, media = graph_candidates_and_stores()
    original = [d.model_dump() for d in candidates]
    mapped = recipe.parent_candidates(candidates)
    sync = recipe.boundary_receipt(resolve(mapped, parents, media))
    asynchronous = recipe.boundary_receipt(await aresolve(mapped, parents, media))
    assert sync == asynchronous
    assert len(sync["parent_ids"]) == 3
    assert sync["image_parts"] == len(sync["media"]) == 2
    assert [d.model_dump() for d in candidates] == original
    assert mapped[1].metadata["_antfly"] == candidates[1].metadata["_antfly"]
    assert all(m is not d for m, d in zip(mapped, candidates, strict=True))


@pytest.mark.parametrize("failure", ["missing_mapping", "missing_parent", "conflicting_mapping"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_graph_parent_failure_stops_at_strict_boundary(failure, asynchronous):
    from examples.multimodal.evidence import IncompleteEvidenceError, aresolve, resolve

    candidates, parents, media = graph_candidates_and_stores()
    parent_id = candidates[1].metadata["_antfly"]["properties"]["parent_id"]
    if failure == "missing_mapping":
        del candidates[1].metadata["_antfly"]["properties"]["parent_id"]
    elif failure == "missing_parent":
        del parents.values[parent_id]
    else:
        candidates[1].metadata["doc_id"] = "different-parent"
    generation_calls = []
    with pytest.raises(IncompleteEvidenceError):
        mapped = recipe.parent_candidates(candidates)
        if asynchronous:
            await aresolve(mapped, parents, media)
        else:
            resolve(mapped, parents, media)
        generation_calls.append("would generate only after complete evidence")
    assert generation_calls == []


@pytest.mark.parametrize("field", ["source_id", "source_sha256"])
def test_graph_mapping_carries_source_provenance_and_rejects_conflicts(field):
    from examples.multimodal.evidence import IncompleteEvidenceError

    candidates, _, _ = graph_candidates_and_stores()
    mapped = recipe.parent_candidates(candidates)
    assert mapped[1].metadata[field] == candidates[1].metadata["_antfly"]["properties"][field]
    assert field not in candidates[1].metadata
    candidates[1].metadata[field] = "conflicting-provenance"
    with pytest.raises(IncompleteEvidenceError, match="provenance"):
        recipe.parent_candidates(candidates)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["swapped_existing_parent", "wrong_source_hash"])
async def test_graph_parent_provenance_failure_blocks_boundary(asynchronous, failure):
    from examples.multimodal.evidence import IncompleteEvidenceError, aresolve, resolve

    candidates, parents, media = graph_candidates_and_stores()
    properties = candidates[1].metadata["_antfly"]["properties"]
    if failure == "swapped_existing_parent":
        properties["parent_id"] = candidates[2].metadata["_antfly"]["properties"]["parent_id"]
    else:
        properties["source_sha256"] = "wrong-hash"
    with pytest.raises(IncompleteEvidenceError, match="disagrees with parent"):
        mapped = recipe.parent_candidates(candidates)
        if asynchronous:
            await aresolve(mapped, parents, media)
        else:
            resolve(mapped, parents, media)
