from __future__ import annotations

import copy
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from antfly import AntflyClient, AntflyNotFoundError
from langchain_core.embeddings import Embeddings
from langgraph.store.base import GetOp, InvalidNamespaceError, ListNamespacesOp, MatchCondition, PutOp, SearchOp
from langgraph_store_antfly import AntflyStore


class MemoryClient:
    def __init__(self):
        self.rows = {}
        self.lock = threading.Lock()
        self.last_query = None

    def create_table(self, table, **kwargs):
        self.rows[table] = {}

    def get_table(self, table):
        if table not in self.rows:
            raise AntflyNotFoundError(404, "missing table")
        return {}

    def get(self, table, key):
        try:
            return copy.deepcopy(self.rows[table][key])
        except KeyError:
            raise AntflyNotFoundError(404, "missing") from None

    def batch(self, table, inserts=None, deletes=None, transforms=None, **kwargs):
        with self.lock:
            data = self.rows[table]
            data.update(copy.deepcopy(inserts or {}))
            for key in deletes or []:
                data.pop(key, None)
            for transform in transforms or []:
                new = transform["key"] not in data
                row = data.setdefault(transform["key"], {})
                for operation in transform["operations"]:
                    if operation["op"] == "$setOnInsert" and not new:
                        continue
                    path = operation["path"][2:].split(".")
                    cursor = row
                    for part in path[:-1]:
                        cursor = cursor.setdefault(part, {})
                    if operation["op"] == "$max" and path[-1] in cursor:
                        cursor[path[-1]] = max(cursor[path[-1]], operation["value"])
                    else:
                        cursor[path[-1]] = copy.deepcopy(operation["value"])
        return {"status": "committed"}

    def request(self, method, path, json):
        table = path.split("/")[4]
        return self.query(table, **json)

    def query(self, table, **kwargs):
        self.last_query = kwargs
        keys = sorted(k for k in self.rows[table] if k.startswith(kwargs.get("filter_prefix", "")))
        if kwargs.get("search_after"):
            assert kwargs["remote_snapshot"] == "snapshot"
            keys = [k for k in keys if k > kwargs["search_after"][0]]
        keys = keys[: kwargs.get("limit", 10)]
        return {
            "responses": [
                {
                    "remote_snapshot": "snapshot",
                    "hits": {
                        "hits": [
                            {"_id": key, "_source": copy.deepcopy(self.rows[table][key]), "_sort": [key]}
                            for key in keys
                        ]
                    },
                }
            ]
        }


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] if "cat" in t else [0.0, 1.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_ordered_batches_namespace_components_and_pagination():
    client = MemoryClient()
    store = AntflyStore.create(client, "mem", page_size=1)
    value = {"nested": {"a": ["é", None]}, "score": 3}
    result = store.batch(
        [
            PutOp(("user", "1"), "a", value),
            GetOp(("user", "1"), "a"),
            PutOp(("user", "10"), "a", {"score": 4}),
            PutOp(("user", "1"), "b", {"score": 1}),
        ]
    )
    assert result[1].value == value
    value["nested"]["a"].append("mutated")
    assert store.get(("user", "1"), "a").value["nested"]["a"] == ["é", None]
    assert [x.key for x in store.search(("user", "1"))] == ["a", "b"]
    assert [x.key for x in store.search(("user", "1"), filter={"score": {"$gte": 2}})] == ["a"]
    assert store.list_namespaces(prefix=("user",), suffix=("1",)) == [("user", "1")]
    assert store.batch([ListNamespacesOp(match_conditions=(MatchCondition("prefix", ("user", "*")),))])[0] == [
        ("user", "1"),
        ("user", "10"),
    ]
    store.delete(("user", "1"), "a")
    assert store.get(("user", "1"), "a") is None


def test_atomic_creation_timestamp_and_index_optout():
    client = MemoryClient()
    store = AntflyStore.create(client, "mem", index={"dims": 2, "embed": TinyEmbeddings(), "fields": ["text"]})
    store.put(("user",), "x", {"text": "cat"})
    created = store.get(("user",), "x").created_at
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda i: store.put(("user",), "x", {"text": f"cat {i}"}), range(12)))
    assert store.get(("user",), "x").created_at == created
    store.put(("user",), "x", {"text": "cat"}, index=False)
    row = next(iter(client.rows["mem"].values()))
    assert row["indexed"] is False
    assert row["_embeddings"]["memory_vectors"] is None


def test_ttl_validation_and_model_free_enumeration():
    store = AntflyStore.create(MemoryClient(), "mem")
    assert store.search(()) == []
    with pytest.raises(AntflyNotFoundError):
        AntflyStore(store.client, "missing").get(("u",), "x")
    store.put(("u",), "typed", {"flag": True, "nested": {"x": [True]}})
    assert store.search(("u",), filter={"flag": 1}) == []
    assert store.search(("u",), filter={"nested": {"x": [1]}}) == []
    assert store.supports_ttl is False
    with pytest.raises(NotImplementedError):
        store.put(("u",), "x", {}, ttl=1)
    with pytest.raises(NotImplementedError):
        store.batch([PutOp(("u",), "x", {}, ttl=1)])
    with pytest.raises(InvalidNamespaceError):
        store.batch([SearchOp(("user.bad",))])
    with pytest.raises(ValueError, match="index configuration"):
        store.search(("u",), query="cat")
    with pytest.raises(ValueError):
        AntflyStore(MemoryClient(), "mem", index={"dims": 2, "embed": TinyEmbeddings(), "typo": True})


@pytest.mark.asyncio
async def test_async_ordered_parity():
    store = AntflyStore.create(MemoryClient(), "mem", page_size=1)
    await store.aput(("u",), "a", {"text": "hello"})
    assert (await store.aget(("u",), "a")).value == {"text": "hello"}
    assert len(await store.asearch(("u",))) == 1
    await store.adelete(("u",), "a")
    assert await store.aget(("u",), "a") is None


@pytest.mark.skipif(not os.environ.get("ANTFLY_TEST_URL"), reason="Requires real Antfly")
def test_live_memory_semantic_filters_concurrent_timestamp_and_reconnect():
    url = os.environ["ANTFLY_TEST_URL"]
    client = AntflyClient(url, token=os.getenv("ANTFLY_TEST_TOKEN"))
    name = "lc_memory_" + uuid.uuid4().hex[:12]
    try:
        store = AntflyStore.create(
            client, name, index={"dims": 2, "embed": TinyEmbeddings(), "fields": ["text"]}, page_size=1, num_shards=2
        )
        store.put(("user", "1"), "a", {"text": "cat", "score": 5})
        store.put(("user", "1"), "b", {"text": "dog", "score": 2})
        store.put(("user", "10"), "c", {"text": "cat", "score": 9})
        created = store.get(("user", "1"), "a").created_at
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(lambda i: store.put(("user", "1"), "a", {"text": "cat", "score": 5, "i": i}), range(8)))
        assert store.get(("user", "1"), "a").created_at == created
        assert len(store.search(("user", "1"))) == 2
        assert [x.key for x in store.search(("user", "1"), query="cat", filter={"score": {"$gte": 3}})] == ["a"]
        assert [x.key for x in store.search(("user", "1"), query="cat", filter={"score": {"$ne": 2}})] == ["a"]
        store.put(("user", "1"), "a", {"text": "cat"}, index=False)
        assert [x.key for x in store.search(("user", "1"), query="cat")] == ["b"]
        with AntflyClient(url, token=os.getenv("ANTFLY_TEST_TOKEN")) as second:
            assert AntflyStore(second, name).get(("user", "1"), "a").value == {"text": "cat"}
    finally:
        client.drop_table(name)
        client.close()


class FieldEmbeddings(Embeddings):
    """Non-unit vectors ensure scores use actual cosine, not dot products."""

    values = {
        "target": [2.0, 0.0],
        "target2": [4.0, 0.0],
        "near": [3.0, 4.0],
        "near2": [6.0, 8.0],
        "unrelated": [0.0, 7.0],
        "opposite": [-5.0, 0.0],
        "query": [10.0, 0.0],
    }

    def __init__(self):
        self.document_inputs = []

    def embed_documents(self, texts):
        self.document_inputs.extend(texts)
        return [self.values[text] for text in texts]

    def embed_query(self, text):
        return self.values[text]


def _populate_multifield(store):
    store.put(("u", "1"), "a", {"title": "unrelated", "body": "target", "group": "keep"})
    store.put(("u", "1"), "b", {"title": "near", "body": "near2", "group": "keep"})
    store.put(("u", "1"), "c", {"title": "opposite", "body": "unrelated", "group": "omit"})
    store.put(("u", "10"), "other", {"title": "target", "body": "target2", "group": "keep"})


def _assert_same_rankings(actual, expected):
    assert [item.key for item in actual] == [item.key for item in expected]
    assert [item.score for item in actual] == pytest.approx([item.score for item in expected], abs=1e-6)


def test_multifield_differential_vs_upstream_with_arrays_and_overrides():
    from langgraph.store.memory import InMemoryStore

    provider = FieldEmbeddings()
    index = {"dims": 2, "embed": provider, "fields": ["title", "body"]}
    store = AntflyStore.create(MemoryClient(), "mem", index=index, page_size=1)
    reference = InMemoryStore(index=index)
    for backend in (store, reference):
        _populate_multifield(backend)
    for kwargs in ({}, {"limit": 1, "offset": 1}, {"filter": {"group": "keep"}}):
        _assert_same_rankings(
            store.search(("u", "1"), query="query", **kwargs), reference.search(("u", "1"), query="query", **kwargs)
        )
    # A single field configuration can still create several vectors via an array
    # or a per-item override; a newly attached client must detect that marker.
    array_index = {"dims": 2, "embed": provider, "fields": ["title"]}
    arrays = AntflyStore.create(MemoryClient(), "arrays", index=array_index, page_size=1)
    array_reference = InMemoryStore(index=array_index)
    for backend in (arrays, array_reference):
        backend.put(("u", "1"), "array", {"title": "opposite", "parts": ["unrelated", "target"]}, index=["parts[*]"])
        backend.put(("u", "1"), "single", {"title": "near"})
    reattached = AntflyStore(arrays.client, "arrays", index=array_index, page_size=1)
    _assert_same_rankings(
        reattached.search(("u", "1"), query="query"), array_reference.search(("u", "1"), query="query")
    )
    assert reattached._has_multifield(()) is True
    assert reattached._has_multifield(("u",)) is True
    assert reattached._has_multifield(("u", "10")) is False
    assert all("\n" not in text for text in provider.document_inputs)
    row = arrays.get(("u", "1"), "array")
    assert row.value["parts"] == ["unrelated", "target"]
    assert len(list(reattached._rows(("u", "1")))) == 2  # marker is not a memory
    arrays.delete(("u", "1"), "array")
    assert arrays._has_multifield(("u", "1"))  # sticky, for cross-process safety


@pytest.mark.parametrize(
    "vector", [[True, 0.0], ["1", 0.0], [float("nan"), 0.0], [float("inf"), 0.0], [0.0, 0.0], [1.0], [10**1000, 0]]
)
def test_invalid_memory_vectors_fail_before_write_or_search(vector):
    class Invalid(Embeddings):
        def embed_documents(self, texts):
            return [vector for _ in texts]

        def embed_query(self, text):
            return vector

    store = AntflyStore.create(MemoryClient(), "mem", index={"dims": 2, "embed": Invalid()})
    with pytest.raises(ValueError):
        store.put(("u",), "a", {"text": "value"})
    assert store.client.rows["mem"] == {}
    with pytest.raises(ValueError):
        store.search(("u",), query="query")


def test_cosine_does_not_assume_unit_norm_or_overflow():
    assert AntflyStore._cosine([1e308, 1e308], [1e308, 0]) == pytest.approx(2**-0.5)
    assert AntflyStore._cosine([1e-308, 0], [1e-308, 0]) == pytest.approx(1)


@pytest.mark.skipif(not os.environ.get("ANTFLY_TEST_URL"), reason="Requires real Antfly")
def test_live_multifield_differential_two_shards_filters_offsets_and_override():
    from langgraph.store.memory import InMemoryStore

    url = os.environ["ANTFLY_TEST_URL"]
    client = AntflyClient(url, token=os.getenv("ANTFLY_TEST_TOKEN"))
    table = "lc_memory_fields_" + uuid.uuid4().hex[:12]
    provider = FieldEmbeddings()
    index = {"dims": 2, "embed": provider, "fields": ["title", "body"]}
    try:
        store = AntflyStore.create(client, table, index=index, page_size=1, num_shards=2)
        reference = InMemoryStore(index=index)
        for backend in (store, reference):
            _populate_multifield(backend)
        for kwargs in ({}, {"limit": 1, "offset": 1}, {"filter": {"group": "keep"}}):
            _assert_same_rankings(
                store.search(("u", "1"), query="query", **kwargs), reference.search(("u", "1"), query="query", **kwargs)
            )
        # Default single-field attachment must still max-pool stored overrides.
        single = {"dims": 2, "embed": provider, "fields": ["title"]}
        with AntflyClient(url, token=os.getenv("ANTFLY_TEST_TOKEN")) as second:
            attached = AntflyStore(second, table, index=single, page_size=1)
            attached.put(("array",), "a", {"title": "opposite", "parts": ["unrelated", "target"]}, index=["parts[*]"])
            attached.put(("array",), "b", {"title": "near"})
            assert [(i.key, round(i.score, 4)) for i in attached.search(("array",), query="query")] == [
                ("a", 1.0),
                ("b", 0.6),
            ]
            # A namespace with no multifield history uses native distances.
            attached.put(("native",), "a", {"title": "near"})
            attached.put(("native",), "b", {"title": "opposite"})
            assert not attached._has_multifield(("native",))
            native = attached.search(("native",), query="query")
            assert [(i.key, round(i.score, 4)) for i in native] == [("a", 0.6), ("b", -1.0)]
            assert [i.key for i in attached.search(("native",), query="query", offset=1, limit=1)] == ["b"]

            async def async_roundtrip():
                import asyncio

                loop = asyncio.get_running_loop()

                async def embed(texts):
                    assert asyncio.get_running_loop() is loop
                    await asyncio.sleep(0)
                    return [FieldEmbeddings.values[text] for text in texts]

                async_store = AntflyStore(
                    second, table, index={"dims": 2, "embed": embed, "fields": ["title", "body"]}, page_size=1
                )
                await async_store.aput(("async",), "a", {"title": "unrelated", "body": "target"})
                await async_store.aput(("async",), "b", {"title": "near", "body": "near2"})
                found = await async_store.asearch(("async",), query="query")
                assert [(i.key, round(i.score, 4)) for i in found] == [("a", 1.0), ("b", 0.6)]
                assert (await async_store.aget(("async",), "a")).value["body"] == "target"

            import asyncio

            asyncio.run(async_roundtrip())
    finally:
        client.drop_table(table)
        client.close()


def test_first_multifield_commit_racing_native_search_rechecks_marker():
    class RacingClient(MemoryClient):
        store = None

        def query(self, table, **kwargs):
            if kwargs.get("embeddings") and self.store is not None:
                store, self.store = self.store, None
                store.put(("u",), "multi", {"parts": ["unrelated", "target"]}, index=["parts[*]"])
                return {"responses": [{"hits": {"hits": []}}]}
            return super().query(table, **kwargs)

    client = RacingClient()
    store = AntflyStore.create(client, "mem", index={"dims": 2, "embed": FieldEmbeddings(), "fields": ["title"]})
    store.put(("u",), "single", {"title": "near"})
    client.store = store
    assert [(i.key, round(i.score, 4)) for i in store.search(("u",), query="query")] == [
        ("multi", 1.0),
        ("single", 0.6),
    ]


def test_explicit_none_index_fields_uses_document_default_without_mutating_config():
    index = {"dims": 2, "embed": TinyEmbeddings(), "fields": None}
    store = AntflyStore.create(MemoryClient(), "mem", index=index)
    store.put(("u",), "a", {"text": "cat"})
    assert store.index["fields"] == ["$"]
    assert index["fields"] is None
    assert next(iter(store.client.rows["mem"].values()))["field_vectors"] == [[1.0, 0.0]]


@pytest.mark.asyncio
async def test_async_callable_uses_caller_loop_and_ordered_ops_without_executor_deadlock():
    import asyncio

    loop = asyncio.get_running_loop()
    lock = asyncio.Lock()
    calls = []

    async def embed(texts):
        assert asyncio.get_running_loop() is loop
        async with lock:
            await asyncio.sleep(0)
            calls.append(list(texts))
            return [FieldEmbeddings.values[text] for text in texts]

    store = AntflyStore.create(
        MemoryClient(), "mem", index={"dims": 2, "embed": embed, "fields": ["title", "body"]}, page_size=1
    )
    result = await store.abatch(
        [
            PutOp(("u",), "a", {"title": "unrelated", "body": "target"}),
            GetOp(("u",), "a"),
            PutOp(("u",), "b", {"title": "near", "body": "near2"}),
            SearchOp(("u",), query="query"),
        ]
    )
    assert result[1].value["body"] == "target"
    assert [(item.key, round(item.score, 4)) for item in result[3]] == [("a", 1.0), ("b", 0.6)]
    assert calls == [["unrelated", "target"], ["near", "near2"], ["query"]]
    # More simultaneous operations than persistence workers must not starve the
    # loop or make workers wait for more work from their own executor.
    await asyncio.wait_for(
        asyncio.gather(
            *(store.aput(("parallel",), str(i), {"title": "unrelated", "body": "target"}) for i in range(24))
        ),
        timeout=10,
    )
    assert len(await store.asearch(("parallel",), query="query", limit=30)) == 24
    assert len(await store.asearch(("parallel",), limit=30)) == 24
    with pytest.raises(ValueError, match="aput/abatch"):
        store.put(("u",), "sync", {"title": "near", "body": "near2"})
    with pytest.raises(ValueError, match="asearch/abatch"):
        store.search(("u",), query="query")
    assert await store.aget(("u",), "sync") is None


def test_pagination_rejects_out_of_order_keys_with_constant_space_guard():
    class Disorder(MemoryClient):
        def query(self, table, **kwargs):
            result = super().query(table, **kwargs)
            result["responses"][0]["hits"]["hits"].reverse()
            return result

    client = Disorder()
    store = AntflyStore.create(client, "mem")
    store.put(("u",), "a", {"text": "a"})
    store.put(("u",), "b", {"text": "b"})
    with pytest.raises(RuntimeError, match="pagination did not advance"):
        store.search(("u",))


def test_timestamp_follows_inference_and_never_moves_backwards():
    entered, release = threading.Event(), threading.Event()

    class Slow(TinyEmbeddings):
        def embed_documents(self, texts):
            if texts == ["slow cat"]:
                entered.set()
                assert release.wait(10)
            return super().embed_documents(texts)

    store = AntflyStore.create(MemoryClient(), "mem", index={"dims": 2, "embed": Slow(), "fields": ["text"]})
    with ThreadPoolExecutor(max_workers=2) as executor:
        delayed = executor.submit(store.put, ("u",), "x", {"text": "slow cat"})
        assert entered.wait(10)
        try:
            store.put(("u",), "x", {"text": "fast cat"})
            first = store.get(("u",), "x")
        finally:
            release.set()
        delayed.result()
    final = store.get(("u",), "x")
    assert final.created_at == first.created_at
    assert final.updated_at >= first.updated_at >= final.created_at

    class DelayedTransport(MemoryClient):
        def batch(self, table, **kwargs):
            transforms = kwargs.get("transforms") or []
            if any(
                operation.get("value") == {"text": "delayed"}
                for transform in transforms
                for operation in transform["operations"]
            ):
                entered.set()
                assert release.wait(10)
            return super().batch(table, **kwargs)

    entered.clear()
    release.clear()
    plain = AntflyStore.create(DelayedTransport(), "plain")
    with ThreadPoolExecutor(max_workers=2) as executor:
        delayed = executor.submit(plain.put, ("u",), "x", {"text": "delayed"})
        assert entered.wait(10)
        try:
            plain.put(("u",), "x", {"text": "fast"})
            first = plain.get(("u",), "x")
        finally:
            release.set()
        delayed.result()
    final = plain.get(("u",), "x")
    assert final.created_at == first.created_at
    assert final.updated_at == first.updated_at
    assert final.updated_at >= final.created_at


@pytest.mark.skipif(not os.environ.get("ANTFLY_TEST_URL"), reason="Requires real Antfly")
def test_live_memory_timestamp_concurrent_inference_and_delayed_transport():
    entered, release = threading.Event(), threading.Event()

    class Slow(TinyEmbeddings):
        def embed_documents(self, texts):
            if texts == ["slow cat"]:
                entered.set()
                assert release.wait(10)
            return super().embed_documents(texts)

    client = AntflyClient(os.environ["ANTFLY_TEST_URL"], token=os.getenv("ANTFLY_TEST_TOKEN"))
    table = "audit_memory_fix_" + uuid.uuid4().hex[:12]
    try:
        store = AntflyStore.create(client, table, index={"dims": 2, "embed": Slow(), "fields": ["text"]}, num_shards=2)
        for delay_transport in (False, True):
            entered.clear()
            release.clear()
            key = "transport" if delay_transport else "inference"
            original_batch = client.batch

            def delayed_batch(*args, _original_batch=original_batch, **kwargs):
                if any(
                    operation.get("value") == {"text": "delayed cat"}
                    for transform in kwargs.get("transforms", [])
                    for operation in transform["operations"]
                ):
                    entered.set()
                    assert release.wait(10)
                return _original_batch(*args, **kwargs)

            if delay_transport:
                client.batch = delayed_batch
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    delayed = executor.submit(
                        store.put, ("u",), key, {"text": "delayed cat" if delay_transport else "slow cat"}
                    )
                    assert entered.wait(10)
                    try:
                        store.put(("u",), key, {"text": "fast cat"})
                        first = store.get(("u",), key)
                    finally:
                        release.set()
                    delayed.result()
                final = store.get(("u",), key)
                assert final.created_at == first.created_at
                assert final.updated_at >= first.updated_at >= final.created_at
                if delay_transport:
                    assert final.updated_at == first.updated_at
            finally:
                client.batch = original_batch
    finally:
        client.drop_table(table)
        client.close()


def _assert_legacy_timestamp_floor(store, seed):
    from datetime import UTC, datetime

    # The legacy clock was ahead of this process; migration must not move time back.
    future = datetime(2099, 1, 1, tzinfo=UTC)
    store.put(("legacy",), "x", {"text": "original"})
    created = store.get(("legacy",), "x").created_at
    seed(future.isoformat())
    assert store.get(("legacy",), "x").updated_at == future
    store.put(("legacy",), "x", {"text": "updated"})
    assert store.get(("legacy",), "x").updated_at == future
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda i: store.put(("legacy",), "x", {"i": i}), range(12)))
    final = store.get(("legacy",), "x")
    assert final.created_at == created
    assert final.updated_at == future
    assert store.search(("legacy",))[0].updated_at == future
    # Once the atomic numeric timestamp exceeds the legacy floor it wins.
    row = {
        "namespace": ["legacy"],
        "key": "x",
        "value": {},
        "created_at": created,
        "updated_at": future.isoformat(),
        "updated_at_us": 4_102_444_800_000_000,
    }
    assert store._item(row).updated_at == datetime(2100, 1, 1, tzinfo=UTC)


def test_legacy_iso_timestamp_survives_migration_and_concurrent_updates():
    client = MemoryClient()
    store = AntflyStore.create(client, "legacy")

    def seed(future):
        row = next(iter(client.rows["legacy"].values()))
        row.pop("updated_at_us")
        row["updated_at"] = future

    _assert_legacy_timestamp_floor(store, seed)


@pytest.mark.skipif(not os.environ.get("ANTFLY_TEST_URL"), reason="Requires real Antfly")
def test_live_legacy_iso_timestamp_survives_migration_and_concurrent_updates():
    from langgraph_store_antfly.store import _key

    client = AntflyClient(os.environ["ANTFLY_TEST_URL"], token=os.getenv("ANTFLY_TEST_TOKEN"))
    table = "audit_memory_fix_legacy_" + uuid.uuid4().hex[:12]
    try:
        store = AntflyStore.create(client, table, num_shards=2)

        def seed(future):
            key = _key(("legacy",), "x")
            row = client.get(table, key)
            row.pop("updated_at_us")
            row["updated_at"] = future
            client.batch(table, inserts={key: row}, sync_level="full_index")

        _assert_legacy_timestamp_floor(store, seed)
    finally:
        client.drop_table(table)
        client.close()
