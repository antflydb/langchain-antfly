from __future__ import annotations

import copy
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from antfly import AntflyClient, AntflyNotFoundError
from langchain_antfly.graphs import AntflyGraphStore
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document


class Client:
    def __init__(self):
        self.rows = {}
        self.versions = {}
        self.last = None
        self.conflicts = 0
        self.indexes = SimpleNamespace(get=lambda *a: {"config": {"type": "graph"}})

    def get_with_version(self, table, key):
        if key not in self.rows:
            raise AntflyNotFoundError(404, "missing")
        return copy.deepcopy(self.rows[key]), str(self.versions[key])

    def request(self, method, path, **kwargs):
        if path.endswith("/query"):
            return self.query("t", **kwargs["json"])
        body = kwargs["json"]
        if self.conflicts:
            self.conflicts -= 1
            return {"status": "aborted"}
        for read in body["read_set"]:
            if str(self.versions.get(read["key"], 0)) != read["version"]:
                return {"status": "aborted"}
        for batch in body["tables"].values():
            self.rows.update(copy.deepcopy(batch.get("inserts", {})))
            for key in batch.get("inserts", {}):
                self.versions[key] = self.versions.get(key, 0) + 1
            for key in batch.get("deletes", []):
                self.rows.pop(key, None)
                self.versions.pop(key, None)
        return {"status": "committed", "tables": {"t": {"status": "committed"}}}

    def query(self, table, **kwargs):
        self.last = copy.deepcopy(kwargs)
        if "graph_queries" in kwargs:
            op = next(iter(kwargs["graph_queries"].values()))
            scope = next(iter(op["match"]["nodes"].values()))["filter"]
            namespace = scope.get("term", (scope.get("conjuncts") or [{}])[0].get("term", ""))
            key = AntflyGraphStore(self, table, namespace=namespace).node_key("Person", "a")
            return {
                "responses": [
                    {
                        "graph_results": {
                            name: {"rows": [{alias: {"key": key} for alias in query["return"]["bindings"]}]}
                            for name, query in kwargs["graph_queries"].items()
                        }
                    }
                ]
            }
        keys = sorted(key for key in self.rows if key.startswith(kwargs.get("filter_prefix", "")))
        if kwargs.get("search_after"):
            assert kwargs["remote_snapshot"] == "snapshot"
            keys = [key for key in keys if key > kwargs["search_after"][0]]
        keys = keys[: kwargs.get("limit", 10)]
        return {
            "responses": [
                {
                    "remote_snapshot": "snapshot",
                    "hits": {"hits": [{"_id": key, "_source": self.rows[key], "_sort": [key]} for key in keys]},
                }
            ]
        }


def document(source="s1", target="b", properties=None):
    a, b = Node(id="a", type="Person", properties=properties or {}), Node(id=target, type="Person")
    return GraphDocument(
        nodes=[a, b],
        relationships=[Relationship(source=a, target=b, type="KNOWS")],
        source=Document(page_content="a knows " + target, metadata={"id": source}),
    )


def match_query(graph):
    return {
        "found": {
            "index": graph.index,
            "match": {
                "anchor": "a",
                "nodes": {"a": {}, "b": {}},
                "edges": [{"from": "a", "to": "b", "types": ["KNOWS"]}],
            },
            "return": {"bindings": ["a", "b"], "limit": 10},
        }
    }


def test_source_owned_graph_union_update_delete_and_occ_retry():
    client = Client()
    graph = AntflyGraphStore(client, "t", namespace="team/1", max_results=1)
    assert graph.node_key("Person", 1) != graph.node_key("Person", "1")
    assert graph.node_key("Person/a", "b") != graph.node_key("Person", "a/b")
    client.conflicts = 1
    graph.add_graph_documents([document(properties={"name": "Alice"})], include_source=True)
    graph.add_graph_documents([document(source="s2", target="c", properties={"title": "Engineer"})])
    graph.add_graph_documents([document(properties={"name": "Updated"})], include_source=True)
    a = client.rows[graph.node_key("Person", "a")]
    assert len(a["_edges"]["graph"]["KNOWS"]) == 2
    assert a["properties"] == {"name": "Updated", "title": "Engineer"}
    graph.refresh_schema()
    assert graph.get_structured_schema["discovery_complete"] is True
    assert "KNOWS" in graph.get_structured_schema["relationship_types"]
    graph.delete_source("s1")
    a = client.rows[graph.node_key("Person", "a")]
    assert a["properties"] == {"title": "Engineer"}
    assert len(a["_edges"]["graph"]["KNOWS"]) == 1
    assert graph.node_key("Person", "b") not in client.rows
    assert graph.node_key("Person", "c") in client.rows


def test_safe_parameter_binding_and_scoped_nodes():
    client = Client()
    graph = AntflyGraphStore(client, "t", namespace="team1")
    query = match_query(graph)
    query["found"]["match"]["nodes"]["a"]["filter"] = {"ids": {"$param": "ids"}}
    key = graph.node_key("Person", 'a"} malicious')
    rows = graph.query(json.dumps(query), {"ids": [key]})
    assert rows[0]["_antfly_graph"]["operation"] == "found"
    nodes = client.last["graph_queries"]["found"]["match"]["nodes"]
    assert nodes["a"]["filter"]["conjuncts"][2]["ids"] == [key]
    assert nodes["b"]["filter"] == {
        "conjuncts": [
            {"term": "team1", "path": "/graph_namespace"},
            {"term": "node", "path": "/graph_kind"},
        ]
    }
    with pytest.raises(ValueError, match="Unused"):
        graph.query(json.dumps(match_query(graph)), {"typo": 1})
    with pytest.raises(ValueError, match="Missing"):
        graph.query(json.dumps(query))
    with pytest.raises(ValueError, match="Cypher"):
        graph.query("MATCH (n) RETURN n")
    with pytest.raises(ValueError):
        graph.traverse([AntflyGraphStore(client, "t", namespace="team10").node_key("Person", "a")])
    other = match_query(graph)
    other["found"]["match"]["nodes"]["a"]["table"] = "other"
    with pytest.raises(ValueError, match="Cross-table"):
        graph.query_graph(other)
    other = match_query(graph)
    other["found"]["match"]["edges"][0]["max_hops"] = 2
    with pytest.raises(ValueError, match="single-hop"):
        graph.query_graph(other)
    with pytest.raises(ValueError, match="arbitrary vector filters"):
        graph.expand_documents([Document(id=key, page_content="x")], filter_query={"term": "secret", "field": "acl"})


@pytest.mark.skipif(not os.environ.get("ANTFLY_TEST_URL"), reason="Requires real Antfly")
def test_live_graph_match_path_source_ownership_and_namespace():
    client = AntflyClient(os.environ["ANTFLY_TEST_URL"], token=os.getenv("ANTFLY_TEST_TOKEN"))
    table = "lc_graph_" + uuid.uuid4().hex[:12]
    try:
        graph = AntflyGraphStore.create(client, table, namespace="team1", max_results=1, num_shards=2)
        graph.add_graph_documents([document()], include_source=True)
        graph.add_graph_documents([document(source="s2", target="c")])
        graph.add_graph_documents([document()], include_source=True)
        assert len(graph.traverse([graph.node_key("Person", "a")], limit=10)) == 1  # configured cap
        graph.max_results = 100
        assert {row["key"] for row in graph.traverse([graph.node_key("Person", "a")])} == {
            graph.node_key("Person", "b"),
            graph.node_key("Person", "c"),
        }
        rows = graph.query_graph(match_query(graph))
        assert len(rows) == 2
        import asyncio

        async_rows = asyncio.run(graph.aquery_graph(match_query(graph)))
        assert {json.dumps(row, sort_keys=True) for row in async_rows} == {
            json.dumps(row, sort_keys=True) for row in rows
        }
        aggregate = match_query(graph)
        aggregate["found"]["return"] = {"aggregates": {"rows": {"count": "*"}}}
        assert graph.query_graph(aggregate)[0]["rows"] == {"value": "2", "exact": True}
        path = graph.query_graph(
            {
                "path": {
                    "shortest_path": {
                        "from": {"key": graph.node_key("Person", "a")},
                        "to": {"key": graph.node_key("Person", "c")},
                    }
                }
            }
        )
        assert path[0]["path"]["length"] == 1
        graph.max_results = 1
        graph.refresh_schema()
        assert graph.get_structured_schema["discovery_complete"]
        second = AntflyGraphStore(client, table, namespace="team10")
        second.add_graph_documents([document()])
        assert len(second.query_graph(match_query(second))) == 1
        graph.max_results = 100
        # An out-of-band relationship must not let hydration or a second hop
        # escape fixed namespace scope, even though its source is authorized.
        z = Node(id="z", type="Person")
        graph.add_graph_documents(
            [
                GraphDocument(
                    nodes=[z], relationships=[], source=Document(page_content="isolated", metadata={"id": "isolated"})
                )
            ]
        )
        owner = client.get(table, graph.node_key("Person", "a"))
        owner["_edges"] = {
            "graph": {
                "KNOWS": [
                    {key: val for key, val in edge.items() if key != "type"}
                    for contribution in owner["contributions"].values()
                    for edge in contribution["edges"]
                ]
                + [{"target": second.node_key("Person", "b")}]
            }
        }
        foreign = client.get(table, second.node_key("Person", "b"))
        foreign["_edges"] = {"graph": {"KNOWS": [{"target": graph.node_key("Person", "z")}]}}
        client.batch(
            table,
            inserts={graph.node_key("Person", "a"): owner, second.node_key("Person", "b"): foreign},
            sync_level="full_index",
        )
        adversarial = graph.traverse([graph.node_key("Person", "a")], max_depth=2)
        assert all(row["document"]["graph_namespace"] == "team1" for row in adversarial)
        assert graph.node_key("Person", "z") not in {row["key"] for row in adversarial}
        assert len(graph.query_graph(match_query(graph))) == 2
        graph.max_results = 100
        graph.delete_source("s1")
        assert [row["key"] for row in graph.traverse([graph.node_key("Person", "a")])] == [
            graph.node_key("Person", "c")
        ]
        assert len(second.query_graph(match_query(second))) == 1
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(
                executor.map(
                    lambda source: graph.add_graph_documents([document(source=source, target=source)]), ["s3", "s4"]
                )
            )
        assert len(graph.traverse([graph.node_key("Person", "a")])) == 3
    finally:
        client.drop_table(table)
        client.close()


def test_result_scope_checks_tables_and_every_intermediate_identity():
    graph = AntflyGraphStore(Client(), "local", namespace="same")
    a, b, c = [graph.node_key("P", key) for key in ("a", "b", "c")]
    foreign = {"table": "foreign", "key": b}
    local = {"key": a}
    end = {"key": c}
    with pytest.raises(RuntimeError, match="escaped"):
        graph._validate_result_scope({"x": foreign}, "match")
    with pytest.raises(RuntimeError, match="escaped"):
        graph._validate_result_scope(
            {"key": c, "depth": 2, "path": [local, foreign, end], "path_edges": []}, "traverse"
        )
    with pytest.raises(RuntimeError, match="escaped"):
        graph._validate_result_scope(
            {"path": {"nodes": [local, foreign, end], "edges": [], "length": 2}}, "shortest_path"
        )
    with pytest.raises(RuntimeError, match="path evidence"):
        graph._validate_result_scope({"key": c, "depth": 2}, "traverse")
    scoped = graph._scoped(match_query(graph))
    assert all(node["table"] == "local" for node in scoped["found"]["match"]["nodes"].values())
    scoped = graph._scoped({"x": {"traverse": {"start": {"keys": [a]}, "include_paths": False}}})
    assert scoped["x"]["traverse"]["include_paths"] is True


@pytest.mark.skipif(not os.environ.get("ANTFLY_TEST_URL"), reason="Requires real Antfly")
def test_live_graph_rejects_foreign_table_hydration_and_returning_paths():
    client = AntflyClient(os.environ["ANTFLY_TEST_URL"], token=os.getenv("ANTFLY_TEST_TOKEN"))
    tables = ["audit_graph_scope_" + uuid.uuid4().hex[:12] for _ in range(2)]
    try:
        local, foreign = [AntflyGraphStore.create(client, table, namespace="same", num_shards=2) for table in tables]
        for store, ids in ((local, ("a", "c")), (foreign, ("b",))):
            store.add_graph_documents(
                [
                    GraphDocument(
                        nodes=[Node(id=key, type="P") for key in ids],
                        relationships=[],
                        source=Document(page_content=store.table),
                    )
                ]
            )
        a, b, c = [local.node_key("P", key) for key in ("a", "b", "c")]
        for store, start, target, target_table in ((local, a, b, foreign.table), (foreign, b, c, local.table)):
            row = client.get(store.table, start)
            row["_edges"] = {"graph": {"LINK": [{"target": target, "metadata": {"target_table": target_table}}]}}
            client.batch(store.table, inserts={start: row}, sync_level="full_index")
        # The terminal document shares the namespace, but belongs to another table.
        with pytest.raises(RuntimeError, match="escaped"):
            local.traverse([a])
        # Reproduce a local terminal reached through a foreign intermediate hop.
        scoped = local._scoped({"q": {"traverse": {"start": {"keys": [a]}, "max_depth": 2, "include_documents": True}}})
        try:
            raw = client.query(local.table, graph_queries=scoped, limit=0)
        except AntflyNotFoundError:
            # Older servers fail closed when routing a cross-table intermediate.
            with pytest.raises(AntflyNotFoundError):
                local.query_graph({"q": {"traverse": {"start": {"keys": [a]}, "max_depth": 2}}})
        else:
            nodes = raw["responses"][0]["graph_results"]["q"]["nodes"]
            assert any(any(hop.get("table") == foreign.table for hop in node["path"]) for node in nodes)
            with pytest.raises(RuntimeError, match="escaped"):
                local.query_graph({"q": {"traverse": {"start": {"keys": [a]}, "max_depth": 2}}})
        # A path query's complete identity evidence cannot cross the table boundary.
        paths = {"q": {"shortest_path": {"from": {"key": a}, "to": {"key": c}}}}
        try:
            raw = client.query(local.table, graph_queries=local._scoped(paths), limit=0)
        except AntflyNotFoundError:
            with pytest.raises(AntflyNotFoundError):
                local.query_graph(paths)
        else:
            returned = raw["responses"][0]["graph_results"]["q"]["paths"]
            assert returned and any(hop.get("table") == foreign.table for hop in returned[0]["path"]["nodes"])
            with pytest.raises(RuntimeError, match="escaped"):
                local.query_graph(paths)
    finally:
        for table in tables:
            client.drop_table(table)
        client.close()


def test_match_rejection_explains_required_server_contract_without_fallback():
    from antfly import AntflyHTTPError

    class Unsupported(Client):
        calls = 0

        def query(self, table, **kwargs):
            self.calls += 1
            assert all(node["table"] == table for node in kwargs["graph_queries"]["found"]["match"]["nodes"].values())
            raise AntflyHTTPError(400, "invalid query request")

    client = Unsupported()
    graph = AntflyGraphStore(client, "t")
    with pytest.raises(RuntimeError, match="table-qualified MATCH aliases"):
        graph.query_graph(match_query(graph))
    assert client.calls == 1


@pytest.mark.parametrize(
    "result",
    [
        {"rows": [{"a": {"key": "foreign", "table": "foreign"}}], "aggregates": {}},
        {"nodes": []},
        {"rows": [{"a": {"key": "missing_b"}}]},
    ],
)
def test_response_projection_cannot_bypass_scope_or_omit_bindings(result):
    class ResultClient(Client):
        def query(self, table, **kwargs):
            return {"graph_results": {"found": result}}

    graph = AntflyGraphStore(ResultClient(), "local", namespace="same")
    with pytest.raises(RuntimeError, match="projection"):
        graph.query_graph(match_query(graph))


@pytest.mark.parametrize(
    "aggregates",
    [
        {},
        {"wrong": {"value": "1", "exact": True}},
        {"total": {"value": 1, "exact": True}},
        {"total": {"value": "1", "exact": False}},
    ],
)
def test_count_projection_requires_requested_names_and_exact_decimal_values(aggregates):
    class ResultClient(Client):
        def query(self, table, **kwargs):
            return {"graph_results": {"found": {"aggregates": aggregates}}}

    graph = AntflyGraphStore(ResultClient(), "local")
    query = match_query(graph)
    query["found"]["return"] = {"aggregates": {"total": {"count": "*"}}}
    with pytest.raises(RuntimeError):
        graph.query_graph(query)


def test_safe_inequality_where_preserves_alias_scope_and_projection():
    graph = AntflyGraphStore(Client(), "local", namespace="team")
    query = match_query(graph)
    expression = {"and": [{"not_equal": {"left": {"alias": "a"}, "right": {"alias": "b"}}}]}
    query["found"]["match"]["where"] = expression
    scoped = graph._scoped(query)
    assert scoped["found"]["match"]["where"] == expression
    assert all(node["table"] == "local" for node in scoped["found"]["match"]["nodes"].values())
    query["found"]["match"]["where"] = {"and": [expression, {"not_exists": {"edges": [{"from": "a", "to": "b"}]}}]}
    with pytest.raises(ValueError, match="WHERE"):
        graph._scoped(query)


def test_graph_expansion_preserves_seed_identity_and_deduplicates_scoped_neighbors(monkeypatch):
    graph = AntflyGraphStore(Client(), "t", namespace="team")
    start = graph.node_key("Person", "a")
    neighbor = graph.node_key("Person", "b")
    seed = Document(id="child-a", page_content="Original seed evidence", metadata={"graph_node_key": start})
    row = {
        "key": neighbor,
        "document": {"properties": {"page_content": "Related evidence"}},
        "_antfly_graph": {"operation": "neighbors"},
        "depth": 1,
    }
    monkeypatch.setattr(
        graph,
        "traverse",
        lambda *args, **kwargs: [
            {"key": start, "document": {"properties": {"page_content": "Do not replace seed"}}},
            row,
            row,
        ],
    )
    documents = graph.expand_documents([seed, seed], limit=2)
    assert documents[0] is seed
    assert [d.id for d in documents] == ["child-a", neighbor]
    assert documents[1].metadata["_antfly"]["graph"]["operation"] == "neighbors"


@pytest.mark.parametrize(
    "responses", [[], None, {}, [None], [{}, {}], [{"graph_results": {"found": {"rows": []}}}] * 2]
)
def test_single_table_graph_response_requires_one_complete_envelope(responses):
    class ResultClient(Client):
        def query(self, table, **kwargs):
            return {"responses": responses}

    graph = AntflyGraphStore(ResultClient(), "local")
    with pytest.raises(RuntimeError, match="exactly one response object"):
        graph.query_graph(match_query(graph))


def test_single_table_graph_response_can_explicitly_return_no_rows():
    class ResultClient(Client):
        def query(self, table, **kwargs):
            return {"responses": [{"graph_results": {"found": {"rows": []}}}]}

    graph = AntflyGraphStore(ResultClient(), "local")
    assert graph.query_graph(match_query(graph)) == []


def test_exact_aggregate_with_truncated_stats_is_not_accepted():
    class ResultClient(Client):
        def query(self, table, **kwargs):
            return {
                "responses": [
                    {
                        "graph_results": {
                            "found": {
                                "aggregates": {"total": {"value": "6", "exact": True}},
                                "stats": {"truncated": True},
                            }
                        }
                    }
                ]
            }

    graph = AntflyGraphStore(ResultClient(), "local")
    query = match_query(graph)
    query["found"]["return"] = {"aggregates": {"total": {"count": "*"}}}
    with pytest.raises(RuntimeError, match="truncated"):
        graph.query_graph(query)


@pytest.mark.parametrize(
    "status,body,limit",
    [
        (200, b"{", 1024),
        (200, b"[]", 1024),
        (200, b"null", 1024),
        (200, b"{}", 1024),
        (200, b'{"status":"committed","tables":[]}', 1024),
        (200, b'{"status":"committed"}', 1024),
        (200, b'{"status":"committed","tables":{"other":{"status":"committed"}}}', 1024),
        (200, b'{"status":"committed"}' + b" " * 64, 32),
        (408, b"timeout", 1024),
        (500, b"internal", 1024),
    ],
)
def test_graph_transaction_unusable_or_ambiguous_reply_is_unknown_without_replay(status, body, limit):
    import httpx
    from langchain_antfly import WriteOutcomeError

    class TransactionClient(AntflyClient):
        def get_with_version(self, table, key):
            raise AntflyNotFoundError(404, "new graph identity")

    calls = []

    def handler(request):
        assert request.url.path == "/db/v1/transactions/commit"
        calls.append(request)
        return httpx.Response(status, content=body)

    client = TransactionClient("http://test", max_json_response_bytes=limit)
    client._client.set_httpx_client(httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)))
    graph = AntflyGraphStore(client, "t", namespace="scope")
    try:
        with pytest.raises(WriteOutcomeError) as caught:
            graph.add_graph_documents([document()])
        assert caught.value.status == "unknown"
        assert caught.value.committed is None
        assert set(caught.value.keys) == {
            graph.node_key("Person", "a"),
            graph.node_key("Person", "b"),
            graph._manifest_key("s1"),
        }
        assert len(calls) == 1
    finally:
        client.close()


def test_graph_transaction_retries_only_explicit_http_occ_abort():
    import httpx

    class TransactionClient(AntflyClient):
        def get_with_version(self, table, key):
            raise AntflyNotFoundError(404, "new graph identity")

    calls = []

    def handler(request):
        calls.append(request)
        return (
            httpx.Response(409, json={"status": "aborted"})
            if len(calls) == 1
            else httpx.Response(200, json={"status": "committed", "tables": {"t": {"status": "committed"}}})
        )

    client = TransactionClient("http://test")
    client._client.set_httpx_client(httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)))
    try:
        AntflyGraphStore(client, "t").add_graph_documents([document()])
        assert len(calls) == 2
    finally:
        client.close()


def test_graph_transaction_preserves_precommit_capacity_rejection():
    from antfly import IndexMutationTemporarilyUnavailableError

    error = IndexMutationTemporarilyUnavailableError("index_probe_unavailable", "precommit")

    class Rejected(Client):
        calls = 0

        def request(self, *args, **kwargs):
            self.calls += 1
            raise error

    client = Rejected()
    with pytest.raises(IndexMutationTemporarilyUnavailableError) as caught:
        AntflyGraphStore(client, "t").add_graph_documents([document()])
    assert caught.value is error
    assert client.calls == 1


@pytest.mark.parametrize("projection", ["binding", "aggregate"])
def test_reserved_graph_provenance_alias_is_rejected_before_network(projection):
    class NoRequest(Client):
        def query(self, *args, **kwargs):
            pytest.fail("Reserved aliases must be rejected before submitting a query")

    graph = AntflyGraphStore(NoRequest(), "t")
    query = match_query(graph)
    if projection == "binding":
        query["found"]["match"] = {"anchor": "_antfly_graph", "nodes": {"_antfly_graph": {}}, "edges": []}
        query["found"]["return"] = {"bindings": ["_antfly_graph"]}
    else:
        query["found"]["return"] = {"aggregates": {"_antfly_graph": {"count": "*"}}}
    with pytest.raises(ValueError, match="_antfly_graph is reserved for result provenance"):
        graph.query_graph(query)


def test_created_graph_schema_indexes_node_kind_and_attach_does_not_mutate_schema():
    class SchemaClient(Client):
        creates = []

        def create_table(self, table, **kwargs):
            self.creates.append((table, kwargs))

    client = SchemaClient()
    graph = AntflyGraphStore.create(client, "owned")
    fields = client.creates[0][1]["schema"]["document_schemas"]["doc"]["schema"]["properties"]
    assert fields["graph_kind"]["x-antfly-types"] == ["keyword"]
    assert fields["graph_namespace"]["x-antfly-types"] == ["keyword"]
    AntflyGraphStore(client, "owned")
    assert len(client.creates) == 1
    assert graph._filter()["conjuncts"][1] == {"term": "node", "path": "/graph_kind"}


@pytest.mark.skipif(not os.getenv("ANTFLY_TEST_GRAPH_URL"), reason="Requires scoped MATCH-capable graph server")
@pytest.mark.parametrize("legacy_schema", [False, True])
def test_live_node_match_and_count_exclude_source_manifests_but_keep_explicit_source_nodes(legacy_schema):
    client = AntflyClient(os.environ["ANTFLY_TEST_GRAPH_URL"], token=os.getenv("ANTFLY_TEST_TOKEN"))
    table = "audit_interface_fix_graph_kind_" + uuid.uuid4().hex[:10]
    try:
        if legacy_schema:
            client.create_table(
                table,
                num_shards=2,
                indexes={
                    "graph": {"type": "graph", "edge_types": []},
                    "graph_predicates": {"type": "algebraic", "derive_from_schema": True},
                },
                schema={
                    "default_type": "doc",
                    "document_schemas": {
                        "doc": {
                            "schema": {
                                "type": "object",
                                "additionalProperties": True,
                                "properties": {"graph_namespace": {"type": "string", "x-antfly-types": ["keyword"]}},
                            }
                        }
                    },
                },
            )
            graph = AntflyGraphStore(client, table, namespace="nodes-only")
        else:
            graph = AntflyGraphStore.create(client, table, namespace="nodes-only", num_shards=2)
        graph.add_graph_documents([document()])
        query = {
            "nodes": {"match": {"anchor": "n", "nodes": {"n": {}}, "edges": []}, "return": {"bindings": ["n"]}},
            "count": {
                "match": {"anchor": "n", "nodes": {"n": {}}, "edges": []},
                "return": {"aggregates": {"total": {"count": "*"}}},
            },
        }
        rows = graph.query_graph(query)
        bindings = [row["n"] for row in rows if "n" in row]
        assert {row["key"] for row in bindings} == {graph.node_key("Person", "a"), graph.node_key("Person", "b")}
        assert next(row["total"] for row in rows if "total" in row) == {"value": "2", "exact": True}
        assert client.get(table, graph._manifest_key("s1"))["graph_kind"] == "source"
        graph.add_graph_documents([document()], include_source=True)
        rows = graph.query_graph(query)
        assert next(row["total"] for row in rows if "total" in row) == {"value": "3", "exact": True}
        assert graph.node_key("__source__", "s1") in {row["n"]["key"] for row in rows if "n" in row}
    finally:
        client.drop_table(table)
        client.close()


@pytest.mark.skipif(not os.getenv("ANTFLY_TEST_GRAPH_URL"), reason="Requires catalog-bound graph selector server")
def test_live_explicit_self_table_selectors_equal_omitted_table_selectors():
    import copy

    client = AntflyClient(os.environ["ANTFLY_TEST_GRAPH_URL"], token=os.getenv("ANTFLY_TEST_TOKEN"))
    table = "audit_state_self_selectors_" + uuid.uuid4().hex[:10]
    try:
        graph = AntflyGraphStore.create(client, table, namespace="self-scope", num_shards=2)
        graph.add_graph_documents([document()])
        a, b = graph.node_key("Person", "a"), graph.node_key("Person", "b")
        walk = {"start": {"identities": [{"key": a}]}, "max_depth": 1, "edge_types": ["KNOWS"]}
        expected_walk = graph.query_graph({"q": {"traverse": walk}})
        assert [row["key"] for row in expected_walk] == [b]
        explicit_walk = copy.deepcopy(walk)
        explicit_walk["start"]["identities"][0]["table"] = table
        assert graph.query_graph({"q": {"traverse": explicit_walk}}) == expected_walk
        for kind in ("shortest_path", "k_shortest_paths"):
            body = {"from": {"key": a}, "to": {"key": b}, "max_depth": 2, "edge_types": ["KNOWS"]}
            if kind == "k_shortest_paths":
                body["k"] = 1
            expected = graph.query_graph({"q": {kind: body}})
            assert len(expected) == 1
            assert expected[0]["path"]["length"] == 1
            assert [node["key"] for node in expected[0]["path"]["nodes"]] == [a, b]
            for endpoints in (("from",), ("to",), ("from", "to")):
                explicit = copy.deepcopy(body)
                for endpoint in endpoints:
                    explicit[endpoint]["table"] = table
                assert graph.query_graph({"q": {kind: explicit}}) == expected
    finally:
        client.drop_table(table)
        client.close()
