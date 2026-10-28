"""Persistent LangGraph memory with component-safe namespaces.

Batches execute sequentially in input order: later reads observe completed earlier
writes. A mixed batch is not a transaction. Cancellation does not roll back writes.
TTL is deliberately unsupported. Clients supplied by callers remain caller-owned.
"""

from __future__ import annotations

import base64
import copy
import inspect
import json
import math
import operator
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from langchain_antfly._utils import hits, request, response_dict, run_async, write
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchItem,
    SearchOp,
    validate_op_namespace,
)
from langgraph.store.base.embed import ensure_embeddings, get_text_at_path


def _encode(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _prefix(namespace: tuple[str, ...]) -> str:
    return "lg1/" + "".join(_encode(part) + "/" for part in namespace)


def _key(namespace: tuple[str, ...], key: str) -> str:
    return _prefix(namespace) + "!" + _encode(key)


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))


def _namespace(namespace: tuple[str, ...]) -> None:
    from langgraph.store.base import InvalidNamespaceError

    validate_op_namespace(GetOp(namespace, ""))
    if not isinstance(namespace, tuple) or not namespace or namespace[0] == "langgraph":
        raise InvalidNamespaceError("Use a nonempty tuple namespace whose root is not 'langgraph'")


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _matches(value: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    for name, condition in (filters or {}).items():
        actual: Any = value
        for part in name.split("."):
            if not isinstance(actual, dict) or part not in actual:
                actual = None
                break
            actual = actual[part]
        if not isinstance(condition, dict) or not any(str(k).startswith("$") for k in condition):
            if not _equal(actual, condition):
                return False
            continue
        for op, expected in condition.items():
            if op not in {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"}:
                raise ValueError(f"Unsupported memory filter operator: {op}")
            try:
                if op not in {"$eq", "$ne"} and (isinstance(actual, bool) or isinstance(expected, bool)):
                    return False
                result = {
                    "$eq": _equal,
                    "$ne": lambda left, right: not _equal(left, right),
                    "$gt": operator.gt,
                    "$gte": operator.ge,
                    "$lt": operator.lt,
                    "$lte": operator.le,
                }[op](actual, expected)
            except TypeError:
                result = False
            if not result:
                return False
    return True


class _PreparedEmbeddings:
    """Per-operation vectors prepared on the caller loop, consumed by a worker."""

    def __init__(self, *, texts=None, vectors=None, query=None, query_vector=None):
        self.texts, self.vectors = texts, vectors
        self.query, self.query_vector = query, query_vector

    def embed_documents(self, texts):
        if texts != self.texts:
            raise ValueError("Memory document inputs changed after async embedding preparation")
        return self.vectors

    def embed_query(self, query):
        if query != self.query:
            raise ValueError("Memory query changed after async embedding preparation")
        return self.query_vector


class AntflyStore(BaseStore):
    """LangGraph long-term memory backed by one Antfly table.

    ``index`` follows LangGraph's IndexConfig: ``dims``, ``embed``, and optional
    ``fields``. Constructors attach; ``create`` provisions a table/index. With no
    index, enumeration and CRUD never load or call an embedding model.

    Each configured field/array element is embedded separately and memory scores
    use maximum cosine similarity. Namespaces that have ever stored multiple
    vectors use an exact paginated scan with client-side pooling (O(namespace
    records) per semantic search). Other namespaces retain native vector top-k.
    Multifield markers are sticky after deletion/update to preserve correctness
    across processes; they affect performance only, never returned memories.
    Unlike InMemoryStore's optional scoreless remainder, semantic queries exclude
    items with indexing disabled; get and no-query search still return them.
    """

    supports_ttl = False
    ttl_config = None

    def __init__(
        self,
        client: Any,
        table: str,
        *,
        index: dict[str, Any] | None = None,
        index_name: str = "memory_vectors",
        page_size: int = 100,
    ):
        if not table or not 1 <= page_size <= 10000:
            raise ValueError("table is required and page_size must be between 1 and 10000")
        if not index_name or "." in index_name:
            raise ValueError("index_name must be nonempty and contain no dots")
        self.client, self.table = client, table
        self.index_name, self.page_size = index_name, page_size
        self.index = dict(index) if index is not None else None
        self.embeddings = None
        self._async_only = False
        if index is not None:
            unknown = set(index) - {"dims", "embed", "fields"}
            if unknown:
                raise ValueError(f"Unknown index options: {sorted(unknown)}")
            if isinstance(index.get("dims"), bool) or not isinstance(index.get("dims"), int) or index["dims"] < 1:
                raise ValueError("index.dims must be a positive integer")
            fields = index.get("fields")
            if fields is not None and (not isinstance(fields, list) or not all(isinstance(f, str) for f in fields)):
                raise ValueError("index.fields must be None or a list of field-path strings")
            self.index["fields"] = ["$"] if fields is None else list(fields)
            embed = index.get("embed")
            self._async_only = inspect.iscoroutinefunction(embed) or (
                callable(embed) and inspect.iscoroutinefunction(embed.__call__)
            )
            self.embeddings = ensure_embeddings(embed)

    @classmethod
    def create(cls, client: Any, table: str, *, num_shards: int | None = None, **kwargs: Any) -> AntflyStore:
        store = cls(client, table, **kwargs)
        indexes = None
        if store.index:
            indexes = {
                store.index_name: {
                    "type": "embeddings",
                    "external": True,
                    "dimension": store.index["dims"],
                    "distance_metric": "cosine",
                }
            }
        client.create_table(table, indexes=indexes, num_shards=num_shards)
        return store

    def _get(self, namespace: tuple[str, ...], key: str) -> dict[str, Any] | None:
        _namespace(namespace)
        try:
            row = response_dict(self.client.get(self.table, _key(namespace, key)))
        except Exception as error:
            if getattr(error, "status_code", None) == 404:
                self.client.get_table(self.table)
                return None
            raise
        if row.get("codec") != "langgraph-memory-v1":
            raise ValueError("Unsupported Antfly memory codec")
        return row

    @staticmethod
    def _item(row: dict[str, Any], score: float | None = None, *, search: bool = False) -> Item:
        legacy = row.get("updated_at")
        updated = datetime.fromisoformat(legacy) if isinstance(legacy, str) else legacy
        if "updated_at_us" in row:
            numeric = datetime.fromtimestamp(row["updated_at_us"] / 1_000_000, UTC)
            # Keep the legacy ISO timestamp as a monotonic floor. New writes
            # deliberately retain it, so migration needs no racy read/upsert.
            updated = max(updated, numeric) if updated is not None else numeric
        args = {
            "namespace": tuple(row["namespace"]),
            "key": row["key"],
            "value": copy.deepcopy(row["value"]),
            "created_at": row["created_at"],
            "updated_at": updated,
        }
        return SearchItem(**args, score=score) if search else Item(**args)

    def _validate_vector(self, vector: Any) -> list[float]:
        assert self.index is not None
        if (
            not isinstance(vector, (list, tuple))
            or len(vector) != self.index["dims"]
            or not all(not isinstance(x, bool) and isinstance(x, (int, float)) for x in vector)
        ):
            raise ValueError("Memory embedding dimension or numeric values are invalid")
        try:
            values = [float(x) for x in vector]
        except (ValueError, OverflowError) as error:
            raise ValueError("Memory embedding values must be finite numbers") from error
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Memory embedding values must be finite numbers")
        if not any(values):
            raise ValueError("Memory embedding vectors must be nonzero")
        return values

    def _vectors(self, texts: list[str]) -> list[list[float]]:
        assert self.embeddings is not None and self.index is not None
        if self._async_only:
            raise ValueError("Async-only embedding callable requires AntflyStore.aput/abatch")
        vectors = self.embeddings.embed_documents(texts)
        if not isinstance(vectors, (list, tuple)) or len(vectors) != len(texts):
            raise ValueError("Embedding provider returned the wrong number of vectors")
        return [self._validate_vector(vector) for vector in vectors]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        # Scale before computing the norm so finite large/small vectors do not
        # overflow/underflow. Never assume the provider returns unit vectors.
        left_scale, right_scale = max(map(abs, left)), max(map(abs, right))
        a, b = [x / left_scale for x in left], [x / right_scale for x in right]
        score = math.fsum(x * y for x, y in zip(a, b, strict=True)) / (math.hypot(*a) * math.hypot(*b))
        return max(-1.0, min(1.0, score))

    @staticmethod
    def _multifield_prefix(namespace: tuple[str, ...]) -> str:
        return _prefix(namespace).replace("lg1/", "lgm1/", 1)

    def _has_multifield(self, namespace: tuple[str, ...]) -> bool:
        # Sticky markers survive process restarts and per-item field overrides.
        # They are stored atomically with the parent, outside the memory keyspace.
        return bool(hits(self.client.query(self.table, filter_prefix=self._multifield_prefix(namespace), limit=1)))

    def _multifield_search(self, op: SearchOp, query_vector: list[float]) -> list[SearchItem]:
        ranked = []
        for row in self._rows(op.namespace_prefix):
            if not row.get("indexed") or not _matches(row["value"], op.filter):
                continue
            vectors = row.get("field_vectors")
            if not vectors:
                raise ValueError("Stored memory predates per-field vectors; re-ingest it before multifield search")
            score = max(self._cosine(query_vector, self._validate_vector(vector)) for vector in vectors)
            ranked.append((score, row))
            # Bound retained candidates while scanning every matching namespace
            # page, so max-pooling is exact without loading the whole corpus.
            ranked.sort(key=lambda item: (-item[0], tuple(item[1]["namespace"]), item[1]["key"]))
            del ranked[op.offset + op.limit :]
        return [self._item(row, score, search=True) for score, row in ranked[op.offset : op.offset + op.limit]]

    def _prepare_put(self, op: PutOp) -> tuple[dict[str, Any] | None, list[str]]:
        _namespace(op.namespace)
        if op.ttl is not None:
            raise NotImplementedError("AntflyStore does not support per-item TTL")
        if op.value is None:
            return None, []
        if (
            op.index is not None
            and op.index is not False
            and (not isinstance(op.index, list) or not all(isinstance(f, str) for f in op.index))
        ):
            raise ValueError("Per-item index must be False or a list of field-path strings")
        value = _json(dict(op.value))
        fields = op.index if isinstance(op.index, list) else (self.index or {}).get("fields", ["$"])
        texts = (
            [text for path in fields for text in get_text_at_path(value, path)]
            if self.index is not None and op.index is not False
            else []
        )
        return value, texts

    def _put(self, op: PutOp) -> None:
        value, texts = self._prepare_put(op)
        key = _key(op.namespace, op.key)
        if value is None:
            write(self.client, self.table, deletes=[key])
            return
        row = {
            "codec": "langgraph-memory-v1",
            "namespace": list(op.namespace),
            "key": op.key,
            "value": value,
            "indexed": False,
            "field_vectors": [],
            "vector_count": 0,
        }
        vector = None
        if texts:
            vectors = self._vectors(texts)
            vector = vectors[0]
            row["indexed"] = True
            row["field_vectors"] = vectors
            row["vector_count"] = len(vectors)
        # Sample after inference, then preserve monotonic update time even if
        # another writer commits while this request is waiting for transport.
        timestamp = datetime.now(UTC)
        now = timestamp.isoformat()
        operations = [
            {"op": "$setOnInsert", "path": "$.created_at", "value": now},
            {
                "op": "$max",
                "path": "$.updated_at_us",
                "value": int(timestamp.timestamp()) * 1_000_000 + timestamp.microsecond,
            },
        ]
        operations.extend({"op": "$set", "path": "$." + name, "value": val} for name, val in row.items())
        if self.index is not None:
            operations.append({"op": "$set", "path": "$._embeddings." + self.index_name, "value": vector})
        transforms = [{"key": key, "upsert": True, "operations": operations}]
        if row["vector_count"] > 1:
            transforms.append(
                {
                    "key": self._multifield_prefix(op.namespace) + "!",
                    "upsert": True,
                    "operations": [{"op": "$set", "path": "$.multifield", "value": True}],
                }
            )
        write(self.client, self.table, transforms=transforms)

    def _rows(self, prefix: tuple[str, ...] = ()) -> Iterable[dict[str, Any]]:
        cursor = snapshot = None
        previous_key = None
        while True:
            args = {
                "filter_prefix": _prefix(prefix),
                "limit": self.page_size,
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
            if not page:
                return
            for hit in page:
                hit = response_dict(hit)
                key = hit.get("_id")
                if not isinstance(key, str) or previous_key is not None and key <= previous_key:
                    raise RuntimeError("Antfly memory pagination did not advance")
                previous_key = key
                row = hit.get("_source", {})
                if row.get("codec") == "langgraph-memory-v1":
                    yield row
            if len(page) < self.page_size:
                return
            cursor = response_dict(page[-1]).get("_sort")
            if cursor is None:
                raise RuntimeError("Antfly did not return the sort cursor required for memory pagination")

    def _search(self, op: SearchOp) -> list[SearchItem]:
        if op.limit < 0 or op.offset < 0:
            raise ValueError("limit and offset must be nonnegative")
        if op.limit == 0:
            return []
        if op.query is None:
            results = []
            matched = 0
            for row in self._rows(op.namespace_prefix):
                if _matches(row["value"], op.filter):
                    if matched >= op.offset:
                        results.append(self._item(row, search=True))
                    matched += 1
                    if len(results) >= op.limit:
                        break
            return results
        if self.index is None or self.embeddings is None:
            raise ValueError("Semantic memory search requires index configuration")
        if self._async_only:
            raise ValueError("Async-only embedding callable requires AntflyStore.asearch/abatch")
        query_vector = self._validate_vector(self.embeddings.embed_query(op.query))
        if self._has_multifield(op.namespace_prefix):
            return self._multifield_search(op, query_vector)
        # Apply JSON comparison filters before vector candidate selection.
        predicates = [{"term": {"path": "/indexed", "value": True}}]
        exclusions = []
        for name, condition in (op.filter or {}).items():
            path = "/value/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in name.split("."))
            if isinstance(condition, dict) and any(str(k).startswith("$") for k in condition):
                for operator, expected in condition.items():
                    if operator == "$eq":
                        predicates.append({"term": {"path": path, "value": expected}})
                    elif operator == "$ne":
                        exclusions.append({"term": {"path": path, "value": expected}})
                    elif operator in {"$gt", "$gte", "$lt", "$lte"}:
                        predicates.append({"range": {path: {operator[1:]: expected}}})
                    else:
                        raise ValueError(f"Unsupported memory filter operator: {operator}")
            else:
                predicates.append({"term": {"path": path, "value": condition}})
        result = self.client.query(
            self.table,
            embeddings={self.index_name: query_vector},
            indexes=[self.index_name],
            filter_prefix=_prefix(op.namespace_prefix),
            filter_query={"conjuncts": predicates},
            exclusion_query={"disjuncts": exclusions} if exclusions else None,
            limit=op.limit + op.offset,
        )
        # A first multifield write can race the initial marker lookup. Since
        # marker and parent commit atomically, checking again after native
        # retrieval prevents returning only its representative field's score.
        if self._has_multifield(op.namespace_prefix):
            return self._multifield_search(op, query_vector)
        selected = []
        for hit in map(response_dict, hits(result)[op.offset : op.offset + op.limit]):
            distance = hit.get("_distance")
            if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance):
                raise ValueError("Antfly memory search did not return a finite cosine distance")
            selected.append(self._item(hit["_source"], 1.0 - distance, search=True))
        return selected

    def _list(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        if op.limit < 0 or op.offset < 0 or op.max_depth is not None and op.max_depth < 1:
            raise ValueError("Invalid namespace listing pagination/depth")
        namespaces = set()
        for row in self._rows():
            ns = tuple(row["namespace"])
            matched = True
            for condition in op.match_conditions or ():
                path = condition.path
                candidate = ns[: len(path)] if condition.match_type == "prefix" else ns[len(ns) - len(path) :]
                if len(candidate) != len(path) or any(
                    a != "*" and a != b for a, b in zip(path, candidate, strict=True)
                ):
                    matched = False
                    break
            if matched:
                namespaces.add(ns[: op.max_depth] if op.max_depth else ns)
        return sorted(namespaces)[op.offset : op.offset + op.limit]

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        """Execute operations sequentially, preserving order; not atomic as a group."""
        operations = list(ops)
        results = []
        for op in operations:
            validate_op_namespace(op)
            if isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            elif isinstance(op, GetOp):
                row = self._get(op.namespace, op.key)
                results.append(None if row is None else self._item(row))
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list(op))
            else:
                raise TypeError(f"Unsupported memory operation: {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        """Prepare embeddings on the caller loop, then persist each op in order.

        No worker waits for an event-loop callback or submits nested work to the
        bounded persistence executor. Async-only providers can safely own loop-
        bound clients/locks; synchronous methods reject those providers clearly.
        """
        results = []
        for original in ops:
            op = copy.deepcopy(original)
            validate_op_namespace(op)
            view = self
            prepared = None
            if self.embeddings is not None:
                if isinstance(op, PutOp):
                    _, texts = self._prepare_put(op)
                    if texts:
                        vectors = await self.embeddings.aembed_documents(texts)
                        prepared = _PreparedEmbeddings(texts=texts, vectors=vectors)
                elif isinstance(op, SearchOp) and op.query is not None and op.limit > 0 and op.offset >= 0:
                    vector = await self.embeddings.aembed_query(op.query)
                    prepared = _PreparedEmbeddings(query=op.query, query_vector=vector)
            if prepared is not None:
                view = copy.copy(self)
                view.embeddings = prepared
                view._async_only = False
            results.extend(await run_async(view.batch, [op]))
        return results
