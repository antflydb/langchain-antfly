"""Frozen supplementary graph evaluation; live execution requires an explicit graph URL."""

import json
import os
import uuid

import pytest
from antfly import AntflyClient
from langchain_antfly import AntflyByteStore, AntflyDocumentStore, AntflyGraphStore
from langchain_core.documents import Document

from examples import evaluate_graph as app
from examples.multimodal.corpus import FIXTURES, PDF, extract
from examples.multimodal.evidence import sha256


class Graph:
    exact = True

    def node_key(self, node_type, identity):
        return f"Evidence/{identity}"

    def traverse(self, keys, **kwargs):
        assert keys == ["Evidence/page-1"]
        assert kwargs["edge_types"] == ["REPORT_CONTEXT"]
        return [{"document": {"graph_id": source}} for source in ("figure-2", "figure-1")]

    def query_graph(self, queries):
        if "count" in queries:
            assert queries["count"]["return"] == {"aggregates": {"relationships": {"count": "*"}}}
            return [{"relationships": {"value": "6", "exact": self.exact}}]
        return [{"path": {"nodes": [{"key": "Evidence/figure-1"}, {"key": "Evidence/figure-2"}], "length": 1}}]


def test_frozen_supplement_evaluates_native_outputs_without_changing_original_labels():
    labels_bytes = app.QUESTIONS.read_bytes()
    assert sha256(labels_bytes) == "f3c9741f9a65eddd49b0055590a0a91b1ee2d0c0b3d748f75e2c301ebd0a83ec"
    original = (FIXTURES / "answers.json").read_bytes()
    labels = json.loads(labels_bytes)
    assert len(labels["questions"]) == 3
    assert {q["operation"] for q in labels["questions"]} == {"traverse", "count", "shortest_path"}
    assert len(json.loads(original)["questions"]) == 20
    result = app.evaluate(Graph(), labels)
    assert all(record["passed"] for record in result)
    assert all(record["actual"] == record["expected"] for record in result)
    assert (FIXTURES / "answers.json").read_bytes() == original


def test_inexact_count_is_failed_evidence_not_a_passing_count():
    graph = Graph()
    graph.exact = False
    result = app.evaluate(graph, json.loads(app.QUESTIONS.read_bytes()))
    assert [record["id"] for record in result if not record["passed"]] == ["graph-exact-context-count"]


def test_cleanup_refuses_input_corpus_tables_and_unowned_tables(monkeypatch):
    class Client:
        def list_tables(self):
            return [{"name": "unowned"}]

        def drop_table(self, table):
            pytest.fail("Unowned table must never be removed")

    class Store:
        def __init__(self, **kwargs):
            pass

        def mget(self, keys):
            return [None]

    monkeypatch.setattr(app, "AntflyDocumentStore", Store)
    with pytest.raises(ValueError, match="corpus table"):
        app.cleanup(Client(), prefix="orchard", graph_table="orchard_parents")
    with pytest.raises(ValueError, match="not owned"):
        app.cleanup(Client(), prefix="orchard", graph_table="unowned")


@pytest.mark.skipif(not os.getenv("ANTFLY_TEST_GRAPH_URL"), reason="Requires scoped MATCH-capable graph server")
def test_live_frozen_graph_supplement_over_persisted_corpus_without_models():
    prefix = "audit_graph_eval_" + uuid.uuid4().hex[:10]
    table = prefix + "_graph"
    client = AntflyClient(os.environ["ANTFLY_TEST_GRAPH_URL"], token=os.getenv("ANTFLY_TEST_TOKEN"))
    try:
        parents, children, media = extract()
        parent_store = AntflyDocumentStore.create(client=client, table=prefix + "_parents")
        byte_store = AntflyByteStore.create(client=client, table=prefix + "_media")
        byte_store.mset(list(media.items()))
        parent_store.mset(
            [(p.id, p) for p in parents]
            + [
                (
                    "ingestion-manifest",
                    Document(
                        page_content="Complete test fixture",
                        metadata={"children": [c.id for c in children], "fingerprint": sha256(PDF.read_bytes())},
                    ),
                )
            ]
        )
        client.create_table(prefix + "_vectors")
        outcome = client.batch(
            prefix + "_vectors",
            inserts={child.id: {"text": child.page_content, "metadata": child.metadata} for child in children},
            sync_level="full_index",
        )
        assert outcome.to_dict()["status"] == "committed"
        report = app.run(client, prefix=prefix, graph_table=table)
        assert report["passed"] and len(report["records"]) == 3
        assert report["model_calls"] == report["generation_calls"] == 0
        assert report["questions_sha256"] == sha256(app.QUESTIONS.read_bytes())
        assert report["fixture_sha256"] == sha256(PDF.read_bytes())
        assert len(report["corpus_sha256"]) == len(report["ingestion_manifest_sha256"]) == 64
        graph = AntflyGraphStore(client, table, namespace=prefix, max_depth=2, max_results=20)
        paths = graph.query_graph(
            {
                "alternatives": {
                    "k_shortest_paths": {
                        "from": {"key": graph.node_key("Evidence", "figure-1")},
                        "to": {"key": graph.node_key("Evidence", "figure-2")},
                        "k": 2,
                        "max_depth": 2,
                        "edge_types": ["REPORT_CONTEXT"],
                    }
                }
            }
        )
        assert [row["path"]["length"] for row in paths] == [1, 2]
        expected_paths = [
            ["figure-1", "figure-2"],
            ["figure-1", "page-1", "figure-2"],
        ]
        assert [[node["key"] for node in row["path"]["nodes"]] for row in paths] == [
            [graph.node_key("Evidence", source) for source in path] for path in expected_paths
        ]
        assert all(node.get("table", table) == table for row in paths for node in row["path"]["nodes"])
        app.cleanup(client, prefix=prefix, graph_table=table)
        assert {item["name"] for item in client.list_tables()} >= {
            prefix + "_parents",
            prefix + "_media",
            prefix + "_vectors",
        }
    finally:
        # All tables here are unique to this test, including failed run output.
        names = {item.get("name", item.get("table_name")) for item in client.list_tables()}
        for name in (table, *(prefix + "_" + suffix for suffix in ("parents", "media", "vectors"))):
            if name in names:
                client.drop_table(name)
        client.close()
