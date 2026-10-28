"""LangChain graph storage using Antfly's canonical JSON dialect, not Cypher.

Each source owns its contribution to shared entity rows. OCC protects concurrent
source updates; removal recomputes the union without deleting other sources' facts.
Property conflicts use lexical source-ID order, with the last source winning.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

try:
    from langchain_community.graphs.graph_store import GraphStore
except ImportError as error:
    raise ImportError("Graph integration requires pip install 'langchain-antfly[graph]'") from error

from ._utils import WriteOutcomeError, check_write_result, hits, request, response_dict, run_async


def _encoded(value: Any) -> str:
    return base64.urlsafe_b64encode(str(value).encode()).decode().rstrip("=")


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _bind(value: Any, params: dict[str, Any], used: set[str]) -> Any:
    if isinstance(value, dict):
        if "$param" in value:
            if set(value) != {"$param"} or not isinstance(value["$param"], str):
                raise ValueError("A parameter placeholder must be exactly {'$param': 'name'}")
            name = value["$param"]
            if name not in params:
                raise ValueError(f"Missing graph parameter: {name}")
            used.add(name)
            return _json(params[name])
        return {key: _bind(item, params, used) for key, item in value.items()}
    if isinstance(value, list):
        return [_bind(item, params, used) for item in value]
    return value


class AntflyGraphStore(GraphStore):
    """Full GraphStore with source-owned ingestion and bounded scoped reads.

    MATCH requires a server supporting explicit table-qualified aliases.
    Traversals require complete path identity evidence; unsupported servers fail
    closed rather than exposing data outside the configured table.

    ``query`` accepts a JSON object keyed by operation names. Placeholder values
    are objects such as ``{"$param": "start"}``, never string interpolation.
    ``query_graph`` accepts the equivalent Python mapping. Use ``node_key`` when
    constructing selectors for ingested LangChain entities.
    """

    def __init__(
        self,
        client: Any,
        table: str,
        index: str = "graph",
        *,
        namespace: str = "",
        max_results: int = 100,
        max_depth: int = 8,
        max_retries: int = 4,
    ):
        if not table or not index or not 1 <= max_results <= 10000 or not 1 <= max_depth <= 64:
            raise ValueError("table/index and bounded positive max_results/max_depth are required")
        self.client, self.table, self.index, self.namespace = client, table, index, namespace
        self.max_results, self.max_depth, self.max_retries = max_results, max_depth, max_retries
        self.prefix = "lcg1/" + _encoded(namespace) + "/"
        self.structured_schema: dict[str, Any] = {"dialect": "antfly-graph-json-v1", "discovery_complete": False}

    @classmethod
    def create(
        cls, client: Any, table: str, index: str = "graph", *, num_shards: int | None = None, **kwargs: Any
    ) -> AntflyGraphStore:
        store = cls(client, table, index, **kwargs)
        client.create_table(
            table,
            num_shards=num_shards,
            indexes={
                index: {"type": "graph", "edge_types": []},
                index + "_predicates": {"type": "algebraic", "derive_from_schema": True},
            },
            schema={
                "default_type": "doc",
                "document_schemas": {
                    "doc": {
                        "schema": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "graph_namespace": {"type": "string", "x-antfly-types": ["keyword"]},
                                "graph_kind": {"type": "string", "x-antfly-types": ["keyword"]},
                            },
                        }
                    }
                },
            },
        )
        return store

    def node_key(self, node_type: str, node_id: str | int) -> str:
        return (
            self.prefix
            + "n/"
            + _encoded(node_type)
            + "/"
            + _encoded(json.dumps(node_id, ensure_ascii=False, separators=(",", ":")))
        )

    @staticmethod
    def source_id(document: Any) -> str:
        identifier = document.metadata.get("id")
        return (
            str(identifier)
            if identifier is not None and str(identifier)
            else hashlib.sha256(document.page_content.encode()).hexdigest()
        )

    def _manifest_key(self, source_id: str) -> str:
        return self.prefix + "s/" + _encoded(source_id)

    def _read(self, key: str) -> tuple[dict[str, Any], str]:
        try:
            value, version = self.client.get_with_version(self.table, key)
            return response_dict(value), version
        except Exception as error:
            if getattr(error, "status_code", None) == 404:
                return {}, "0"
            raise

    def _update_source(self, source: str, contributions: dict[str, dict[str, Any]] | None) -> None:
        manifest_key = self._manifest_key(source)
        for _ in range(self.max_retries):
            manifest, version = self._read(manifest_key)
            old_keys = set(manifest.get("nodes", []))
            new_keys = set(contributions or {})
            keys = sorted(old_keys | new_keys)
            if len(keys) > 4095:
                raise ValueError("One graph source may own at most 4095 nodes per atomic ingestion")
            read_set = [{"table": self.table, "key": manifest_key, "version": version}]
            inserts, deletes = {}, []
            for key in keys:
                row, row_version = self._read(key)
                read_set.append({"table": self.table, "key": key, "version": row_version})
                owners = copy.deepcopy(row.get("contributions", {}))
                owner = hashlib.sha256(source.encode()).hexdigest()
                if contributions and key in contributions:
                    owners[owner] = {**contributions[key], "source_id": source}
                else:
                    owners.pop(owner, None)
                if not owners:
                    deletes.append(key)
                    continue
                merged, edges = {}, defaultdict(list)
                for _, contribution in sorted(owners.items(), key=lambda item: item[1]["source_id"]):
                    merged.update(contribution.get("properties", {}))
                    for edge in contribution.get("edges", []):
                        edges[edge["type"]].append({k: v for k, v in edge.items() if k != "type"})
                exemplar = next(iter(owners.values()))
                inserts[key] = {
                    "graph_kind": "node",
                    "graph_namespace": self.namespace,
                    "graph_type": exemplar["type"],
                    "graph_id": exemplar["id"],
                    "properties": merged,
                    "contributions": owners,
                    "_edges": {self.index: dict(edges)},
                }
            if contributions is None:
                deletes.append(manifest_key)
            else:
                inserts[manifest_key] = {
                    "graph_kind": "source",
                    "graph_namespace": self.namespace,
                    "source_id": source,
                    "nodes": sorted(new_keys),
                }
            try:
                raw_result = request(
                    self.client,
                    "POST",
                    "/db/v1/transactions/commit",
                    json={
                        "read_set": read_set,
                        "tables": {self.table: {"inserts": inserts, "deletes": deletes}},
                        "sync_level": "full_index",
                    },
                )
            except Exception as error:
                # A 409 carrying an explicit aborted transaction is safe to
                # recompute from fresh versions. Other failures may be committed.
                if getattr(error, "status_code", None) == 409:
                    try:
                        rejected = json.loads(getattr(error, "detail", ""))
                    except (TypeError, ValueError):
                        rejected = {}
                    if isinstance(rejected, dict) and rejected.get("status") == "aborted":
                        continue
                import httpx
                from antfly import AntflyHTTPError, AntflyResponseError

                if isinstance(error, (httpx.TransportError, OSError, TimeoutError, AntflyResponseError)) or (
                    isinstance(error, AntflyHTTPError) and (error.status_code == 408 or error.status_code >= 500)
                ):
                    raise WriteOutcomeError("unknown", tuple(inserts) + tuple(deletes), str(error)) from error
                raise
            try:
                result = response_dict(raw_result)
            except (ValueError, TypeError, AttributeError) as error:
                raise WriteOutcomeError(
                    "unknown", tuple(inserts) + tuple(deletes), "Unusable graph transaction response"
                ) from error
            if result.get("status") == "aborted":
                continue  # Only explicit OCC aborts can be retried safely.
            table_results = result.get("tables")
            if not isinstance(table_results, dict) or set(table_results) != {self.table}:
                raise WriteOutcomeError(
                    "unknown", tuple(inserts) + tuple(deletes), "Unusable graph transaction table results"
                )
            for table_result in table_results.values():
                check_write_result(table_result, keys=tuple(inserts) + tuple(deletes))
            check_write_result(result, keys=tuple(inserts) + tuple(deletes))
            return
        raise RuntimeError("Graph ingestion conflicted repeatedly; no uncommitted attempt was applied")

    def add_graph_documents(self, graph_documents: list[Any], include_source: bool = False) -> None:
        for graph in graph_documents:
            source = self.source_id(graph.source)
            contributions: dict[str, dict[str, Any]] = {}
            for node in list(graph.nodes) + [
                endpoint for edge in graph.relationships for endpoint in (edge.source, edge.target)
            ]:
                key = self.node_key(node.type, node.id)
                contribution = contributions.setdefault(
                    key, {"type": node.type, "id": node.id, "properties": {}, "edges": []}
                )
                contribution["properties"].update(_json(node.properties))
            for edge in graph.relationships:
                start, target = (
                    self.node_key(edge.source.type, edge.source.id),
                    self.node_key(edge.target.type, edge.target.id),
                )
                properties = _json(edge.properties)
                identity = json.dumps([source, start, target, edge.type, properties.get("id", "")], sort_keys=True)
                relationship = {
                    "target": target,
                    "type": edge.type,
                    "edge_id": hashlib.sha256(identity.encode()).hexdigest(),
                    "weight": 1.0,
                    "metadata": {**properties, "source_id": source},
                }
                if relationship not in contributions[start]["edges"]:
                    contributions[start]["edges"].append(relationship)
            if include_source:
                source_key = self.node_key("__source__", source)
                contributions[source_key] = {
                    "type": "__source__",
                    "id": source,
                    "properties": {"page_content": graph.source.page_content, "metadata": _json(graph.source.metadata)},
                    "edges": [
                        {
                            "target": key,
                            "type": "MENTIONS",
                            "weight": 1.0,
                            "edge_id": hashlib.sha256((source + key).encode()).hexdigest(),
                            "metadata": {"source_id": source},
                        }
                        for key in sorted(contributions)
                    ],
                }
            self._update_source(source, contributions)

    def delete_source(self, source_id: str) -> None:
        """Remove one source's facts, preserving every other source contribution."""
        self._update_source(str(source_id), None)

    async def aadd_graph_documents(self, graph_documents: list[Any], include_source: bool = False) -> None:
        await run_async(self.add_graph_documents, graph_documents, include_source)

    @property
    def get_schema(self) -> str:
        return json.dumps(self.structured_schema, ensure_ascii=False, sort_keys=True)

    @property
    def get_structured_schema(self) -> dict[str, Any]:
        return copy.deepcopy(self.structured_schema)

    def refresh_schema(self) -> None:
        types, properties, relationships = set(), defaultdict(set), set()
        cursor, snapshot, complete, seen = None, None, False, set()
        while True:
            args = {
                "filter_prefix": self.prefix + "n/",
                "limit": self.max_results,
                "order_by": [{"field": "_id", "desc": False}],
            }
            if cursor is not None:
                args["search_after"] = cursor
                args["remote_snapshot"] = snapshot
            response = response_dict(
                request(self.client, "POST", f"/db/v1/tables/{quote(self.table, safe='')}/query", json=args)
            )
            page = hits(response)
            snapshot = response.get("responses", [response])[0].get("remote_snapshot", snapshot)
            for hit in page:
                key, row = hit["_id"], hit["_source"]
                if key in seen:
                    raise RuntimeError("Graph schema pagination did not advance")
                seen.add(key)
                types.add(row["graph_type"])
                properties[row["graph_type"]].update(row.get("properties", {}))
                for contribution in row.get("contributions", {}).values():
                    relationships.update(edge["type"] for edge in contribution.get("edges", []))
            if len(page) < self.max_results:
                complete = True
                break
            cursor = page[-1].get("_sort")
            if cursor is None:
                break
        self.structured_schema = {
            "dialect": "antfly-graph-json-v1",
            "namespace": self.namespace,
            "node_types": sorted(types),
            "node_properties": {key: sorted(value) for key, value in properties.items()},
            "relationship_types": sorted(relationships),
            "discovery_complete": complete,
            "discovery": "stored source contributions; not a transactional schema snapshot",
            "declared": response_dict(self.client.indexes.get(self.table, self.index)).get("config", {}),
        }

    def _filter(self, existing: Any = None) -> dict[str, Any]:
        clauses = [
            {"term": self.namespace, "path": "/graph_namespace"},
            {"term": "node", "path": "/graph_kind"},
        ]
        if existing:
            clauses.append(existing)
        return {"conjuncts": clauses}

    def _endpoint(self, endpoint: dict[str, Any]) -> None:
        if endpoint.get("table", self.table) != self.table or not str(endpoint.get("key", "")).startswith(
            self.prefix + "n/"
        ):
            raise ValueError("Graph endpoint is outside this store's fixed table/namespace")

    @staticmethod
    def _validate_where(where: Any, aliases: set[str], depth: int = 0) -> None:
        if not isinstance(where, dict) or depth > 32:
            raise ValueError("Scoped MATCH requires a bounded WHERE expression")
        if set(where) == {"and"} and isinstance(where["and"], list) and where["and"]:
            for child in where["and"]:
                AntflyGraphStore._validate_where(child, aliases, depth + 1)
            return
        if set(where) == {"not_equal"}:
            comparison = where["not_equal"]
            if isinstance(comparison, dict) and set(comparison) == {"left", "right"}:
                operands = [comparison[side] for side in ("left", "right")]
                if all(
                    isinstance(item, dict)
                    and set(item) == {"alias"}
                    and isinstance(item["alias"], str)
                    and item["alias"] in aliases
                    for item in operands
                ):
                    return
        raise ValueError("Scoped MATCH supports only alias not_equal and conjunctions in WHERE")

    def _scoped(self, queries: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(queries, Mapping) or not queries or len(queries) > 8:
            raise ValueError("Provide between 1 and 8 named graph operations")
        queries = _json(queries)
        for name, op in queries.items():
            if not isinstance(name, str) or not name or not isinstance(op, dict):
                raise ValueError("Graph operations require nonempty names and objects")
            kind = set(op) & {"match", "traverse", "shortest_path", "k_shortest_paths"}
            if len(kind) != 1 or set(op) - {"index", "return", *kind}:
                raise ValueError("Invalid canonical graph operation")
            if op.get("index", self.index) != self.index:
                raise ValueError("Graph index is fixed by the store")
            op["index"] = self.index
            kind = next(iter(kind))
            body = op[kind]
            if kind == "match":
                if not body.get("nodes") or body.get("anchor") not in body["nodes"]:
                    raise ValueError("MATCH requires declared nodes and anchor")
                if "_antfly_graph" in body["nodes"]:
                    raise ValueError("Graph alias _antfly_graph is reserved for result provenance")
                for node in body["nodes"].values():
                    if node.get("table", self.table) != self.table:
                        raise ValueError("Cross-table MATCH is outside the store scope")
                    node["table"] = self.table
                    node["filter"] = self._filter(node.get("filter"))
                if any(edge.get("max_hops", 1) != 1 for edge in body.get("edges", [])):
                    raise ValueError("Scoped MATCH requires explicit single-hop aliases for every node")
                if body.get("optional"):
                    raise ValueError("Scoped MATCH optional patterns require explicit reviewed support")
                if body.get("where") is not None:
                    self._validate_where(body["where"], set(body["nodes"]))
                projection = op.setdefault("return", {"bindings": list(body["nodes"])})
                if "_antfly_graph" in projection.get("bindings", ()) or "_antfly_graph" in projection.get(
                    "aggregates", {}
                ):
                    raise ValueError("Graph alias _antfly_graph is reserved for result provenance")
                if "aggregates" not in projection:
                    projection["limit"] = min(projection.get("limit", self.max_results), self.max_results)
            else:
                body["filter"] = self._filter(body.get("filter"))
                body["max_depth"] = min(body.get("max_depth", self.max_depth), self.max_depth)
                if kind == "traverse":
                    selector = body.get("start", {})
                    if set(selector) == {"keys"}:
                        for key in selector["keys"]:
                            self._endpoint({"key": key})
                    elif set(selector) == {"identities"}:
                        for endpoint in selector["identities"]:
                            self._endpoint(endpoint)
                    else:
                        raise ValueError("Scoped traversal requires explicit keys or identities")
                    body["limit"] = min(body.get("limit", self.max_results), self.max_results)
                    # Terminal identity alone cannot detect an A → foreign B → A path.
                    body["include_paths"] = True
                else:
                    self._endpoint(body.get("from", {}))
                    self._endpoint(body.get("to", {}))
                    if kind == "k_shortest_paths":
                        body["k"] = min(body.get("k", 1), self.max_results)
        return queries

    def _validate_result_scope(self, row: dict[str, Any], kind: str) -> None:
        """Fail closed on returned identities, including every intermediate hop."""

        def endpoint(value: Any) -> None:
            if not isinstance(value, dict):
                raise RuntimeError("Antfly graph response omitted a scoped identity")
            try:
                self._endpoint(value)
            except ValueError as error:
                raise RuntimeError("Antfly graph response escaped this store's table/namespace") from error
            document = value.get("document")
            if isinstance(document, dict) and "graph_namespace" in document:
                if document["graph_namespace"] != self.namespace:
                    raise RuntimeError("Antfly graph document escaped this store's namespace")

        def edges(values: Any) -> None:
            if not isinstance(values, list):
                raise RuntimeError("Antfly graph response omitted path edge evidence")
            for edge in values:
                endpoint(edge.get("from"))
                endpoint(edge.get("to"))

        if kind == "match":
            for binding in row.values():
                if binding is not None:
                    endpoint(binding)
        elif kind == "traverse":
            endpoint(row)
            path = row.get("path")
            if not isinstance(path, list) or not path or len(path) != row.get("depth", -1) + 1:
                raise RuntimeError("Antfly graph traversal omitted complete path evidence")
            for node in path:
                endpoint(node)
            edges(row.get("path_edges"))
        else:
            path = row.get("path", {})
            nodes = path.get("nodes")
            if not isinstance(nodes, list) or not nodes or len(nodes) != path.get("length", -1) + 1:
                raise RuntimeError("Antfly graph path omitted complete identity evidence")
            for node in nodes:
                endpoint(node)
            edges(path.get("edges"))
            if "document" in row:
                endpoint({**nodes[-1], "document": row["document"]})

    def query_graph(self, queries: Mapping[str, Any]) -> list[dict[str, Any]]:
        scoped = self._scoped(queries)
        try:
            result = response_dict(self.client.query(self.table, graph_queries=scoped, limit=0))
        except Exception as error:
            if getattr(error, "status_code", None) == 400 and any("match" in op for op in scoped.values()):
                raise RuntimeError(
                    "Antfly rejected the scoped MATCH query. The server must support explicit "
                    "table-qualified MATCH aliases; unqualified fallback is unsafe."
                ) from error
            raise
        responses = result.get("responses", [result])
        if not isinstance(responses, list) or len(responses) != 1 or not isinstance(responses[0], dict):
            raise RuntimeError("Antfly graph query requires exactly one response object for its table")
        rows = []
        for response in responses:
            if response.get("status", 200) != 200 or response.get("error"):
                raise RuntimeError("Antfly graph query returned an unsuccessful response")
            results = response.get("graph_results", {})
            if not isinstance(results, dict) or set(results) != set(scoped):
                raise RuntimeError("Antfly graph response is missing requested operations")
            for name, value in results.items():
                if not isinstance(value, dict):
                    raise RuntimeError(f"Graph operation {name!r} returned an invalid result object")
                if value.get("status") not in (None, "completed", "complete", "success", 200):
                    raise RuntimeError(f"Graph operation {name!r} failed")
                if value.get("complete") is False:
                    raise RuntimeError(f"Graph operation {name!r} returned incomplete evidence")
                kind = next(iter(set(scoped[name]) & {"match", "traverse", "shortest_path", "k_shortest_paths"}))
                projection = scoped[name].get("return", {})
                aggregate = kind == "match" and "aggregates" in projection
                field = (
                    "aggregates"
                    if aggregate
                    else "rows"
                    if kind == "match"
                    else "nodes"
                    if kind == "traverse"
                    else "paths"
                )
                result_kind = {"aggregates": "aggregates", "rows": "bindings", "nodes": "nodes", "paths": "paths"}[
                    field
                ]
                if (
                    set(value) & {"rows", "nodes", "paths", "aggregates"} != {field}
                    or value.get("kind", result_kind) != result_kind
                ):
                    raise RuntimeError("Antfly graph result does not match the requested projection")
                projected = value[field]
                if aggregate:
                    stats = value.get("stats", {})
                    if not isinstance(stats, dict) or stats.get("truncated", False) is not False:
                        raise RuntimeError("Antfly graph aggregate returned truncated or invalid completeness evidence")
                    if not isinstance(projected, dict) or set(projected) != set(projection["aggregates"]):
                        raise RuntimeError("Antfly graph aggregate projection is incomplete or mismatched")
                    for item in projected.values():
                        if (
                            not isinstance(item, dict)
                            or item.get("exact") is not True
                            or not isinstance(item.get("value"), str)
                            or not item["value"].isascii()
                            or not item["value"].isdecimal()
                        ):
                            raise RuntimeError("Antfly graph aggregates are not exact decimal counts")
                    projected = [projected]
                elif not isinstance(projected, list):
                    raise RuntimeError("Antfly graph result omitted its projected result list")
                for row in projected:
                    if not isinstance(row, dict):
                        raise RuntimeError("Antfly graph result contains an invalid row")
                    if not aggregate:
                        if kind == "match" and (
                            set(row) != set(projection["bindings"]) or any(value is None for value in row.values())
                        ):
                            raise RuntimeError("Antfly graph binding projection is incomplete or mismatched")
                        self._validate_result_scope(row, kind)
                    rows.append(
                        {
                            **row,
                            "_antfly_graph": {
                                "operation": name,
                                "truncated": value.get("stats", {}).get("truncated", False),
                            },
                        }
                    )
        return rows

    def query(self, query: str, params: dict | None = None) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(query)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("Graph query must be canonical Antfly JSON; Cypher is unsupported") from error
        params, used = params or {}, set()
        bound = _bind(parsed, params, used)
        if set(params) != used:
            raise ValueError(f"Unused graph parameters: {sorted(set(params) - used)}")
        return self.query_graph(bound)

    def traverse(
        self, keys: Sequence[str], *, max_depth: int = 1, limit: int = 20, edge_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        body = {"start": {"keys": list(keys)}, "max_depth": max_depth, "limit": limit, "include_documents": True}
        if edge_types is not None:
            body["edge_types"] = edge_types
        return self.query_graph({"neighbors": {"index": self.index, "traverse": body}})

    async def aquery_graph(self, queries: Mapping[str, Any]) -> list[dict[str, Any]]:
        return await run_async(self.query_graph, queries)

    def expand_documents(
        self,
        documents: Sequence[Any],
        *,
        seed_keys: Any = None,
        max_depth: int = 1,
        limit: int = 20,
        edge_types: list[str] | None = None,
        filter_query: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Preserve retrieved seeds and append bounded unique graph neighbors.

        Seed identity mapping must be explicit; neighbor rows matching seed graph
        keys are omitted so expansion enriches rather than replaces evidence.
        """
        from langchain_core.documents import Document

        if filter_query:
            raise ValueError("Graph expansion cannot enforce arbitrary vector filters; use a scoped graph query")
        keys = (
            list(seed_keys(documents))
            if callable(seed_keys)
            else [document.metadata.get("graph_node_key", document.id) for document in documents]
        )
        if not keys or any(key is None for key in keys):
            raise ValueError("Graph expansion requires graph_node_key metadata or a seed_keys mapper")
        rows = self.traverse(keys, max_depth=max_depth, limit=limit, edge_types=edge_types)
        result = []
        document_ids = set()
        for document in documents:
            identity = document.id if document.id is not None else id(document)
            if identity not in document_ids:
                result.append(document)
                document_ids.add(identity)
        seen = set(keys)
        for row in rows:
            if row["key"] in seen:
                continue
            seen.add(row["key"])
            value = row.get("document")
            if not isinstance(value, dict):
                raise RuntimeError("Graph expansion returned a node without its requested document")  # noqa: TRY004
            properties = value.get("properties", value)
            content = properties.get("page_content", properties.get("text", properties.get("body")))
            if not isinstance(content, str):
                content = json.dumps(properties, ensure_ascii=False, sort_keys=True)
            result.append(
                Document(
                    id=row["key"],
                    page_content=content,
                    metadata={
                        "_antfly": {
                            "graph": row.get("_antfly_graph"),
                            "depth": row.get("depth"),
                            "properties": properties,
                        }
                    },
                )
            )
        return result
