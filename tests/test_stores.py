import asyncio
from copy import deepcopy

import pytest
from antfly import AntflyHTTPError, AntflyNotFoundError
from langchain_antfly import AntflyByteStore, AntflyDocumentStore, AntflyJSONStore, WriteOutcomeError
from langchain_antfly._utils import check_write_result, run_async
from langchain_core.documents import Document


class MemoryClient:
    def __init__(self):
        self.data = {}
        self.calls = 0
        self.fail_after = None

    def create_table(self, table):
        self.data[table] = {}

    def get_table(self, table):
        if table not in self.data:
            raise AntflyNotFoundError(404, "table missing")
        return {}

    def get(self, table, key):
        self.get_table(table)
        if key not in self.data[table]:
            raise AntflyNotFoundError(404, "key missing")
        return deepcopy(self.data[table][key])

    def batch(self, table, inserts=None, deletes=None, **kwargs):
        self.calls += 1
        if self.fail_after == self.calls:
            return {"status": "committed_repair_required", "failure": {"retryable": False}}
        self.data[table].update(deepcopy(inserts or {}))
        for k in deletes or []:
            self.data[table].pop(k, None)
        return {"status": "committed"}

    def query(self, table, filter_prefix, limit, search_after, **kwargs):
        after = search_after[0] if search_after else ""
        keys = sorted(k for k in self.data[table] if k.startswith(filter_prefix) and (not search_after or k > after))[
            :limit
        ]
        return {"responses": [{"hits": {"hits": [{"_id": k, "_sort": [k]} for k in keys]}}]}


@pytest.mark.parametrize(
    "cls,value",
    [
        (AntflyJSONStore, {"nested": ["é", None, True]}),
        (AntflyDocumentStore, Document(id="source", page_content="chart", metadata={"page": 2})),
        (AntflyByteStore, b"\x00\xffimage"),
    ],
)
def test_roundtrip_prefix_and_missing(cls, value):
    c = MemoryClient()
    s = cls.create(client=c, table="x", namespace="user/1", page_size=1)
    s.mset([("é/a", value), ("é/b", value), ("", value)])
    assert s.mget(["é/a", "missing", "é/a"]) == [value, None, value]
    assert list(s.yield_keys(prefix="é/")) == ["é/a", "é/b"]
    other = cls(client=c, table="x", namespace="user/10")
    assert other.mget(["é/a"]) == [None]
    assert list(other.yield_keys()) == []
    s.mdelete(["é/a", "missing"])
    assert s.mget(["é/a"]) == [None]


def test_wrong_codec_invalid_and_size_fail_before_mutation():
    c = MemoryClient()
    s = AntflyByteStore.create(client=c, table="x", max_value_bytes=2)
    with pytest.raises(ValueError):
        s.mset([("a", b"x"), ("b", b"abc")])
    assert c.calls == 0
    s.mset([("a", b"x")])
    j = AntflyJSONStore(client=c, table="x")
    with pytest.raises(ValueError):
        j.mget(["a"])
    with pytest.raises(AntflyNotFoundError):
        AntflyJSONStore(client=c, table="missing").mget(["x"])

    def forbidden(*args):
        raise AntflyHTTPError(403, "denied")

    c.get = forbidden
    with pytest.raises(AntflyHTTPError):
        s.mget(["a"])


def test_partial_commit_and_pending_are_observable():
    c = MemoryClient()
    s = AntflyJSONStore.create(client=c, table="x", batch_size=1)
    c.fail_after = 2
    with pytest.raises(WriteOutcomeError) as caught:
        s.mset([("a", 1), ("b", 2)])
    assert caught.value.committed_keys == ("a",)
    assert caught.value.status == "committed_repair_required"
    assert c.calls == 2
    with pytest.raises(WriteOutcomeError):
        check_write_result({"status": "committed_pending"})
    assert check_write_result({"status": "committed_pending"}, wait=False)


@pytest.mark.asyncio
async def test_async_parity_and_bounded_worker():
    c = MemoryClient()
    s = AntflyJSONStore.create(client=c, table="x", page_size=1)
    await s.amset([("a", 1), ("b", 2)])
    assert await s.amget(["a", "no"]) == [1, None]
    assert [k async for k in s.ayield_keys()] == ["a", "b"]
    await s.amdelete(["a"])
    assert await s.amget(["a"]) == [None]
    assert await asyncio.gather(*(run_async(lambda x: x + 1, i) for i in range(32))) == list(range(1, 33))


def test_byte_bounded_chunks_and_failed_second_chunk_are_not_replayed():
    from langchain_antfly._utils import write

    c = MemoryClient()
    c.max_write_request_bytes = 170
    c.create_table("x")
    rows = {str(i): {"text": "a" * 55} for i in range(4)}
    write(c, "x", inserts=rows)
    assert c.calls == 4
    assert c.data["x"] == rows
    c.calls = 0
    c.fail_after = 2
    with pytest.raises(WriteOutcomeError) as error:
        write(c, "x", inserts=rows)
    assert c.calls == 2
    assert error.value.committed_keys == ("0",)
    c.calls = 0
    with pytest.raises(ValueError):
        write(c, "x", inserts={"large": {"text": "x" * 300}})
    assert c.calls == 0


def test_query_failure_and_unknown_write_outcomes_fail_closed():
    from langchain_antfly._utils import hits

    with pytest.raises(RuntimeError):
        hits({"responses": [{"status": 503, "error": "busy"}]})
    with pytest.raises(WriteOutcomeError) as error:
        check_write_result({"status": "committed_visibility_pending"})
    assert error.value.committed is True
    assert WriteOutcomeError("unknown").committed is None


@pytest.mark.parametrize("operation", ["mset", "mdelete"])
def test_nested_chunk_failure_reports_all_commits_as_logical_keys(operation):
    from langchain_antfly._utils import WriteOutcomeError

    class Client:
        max_write_request_bytes = 330 if operation == "mset" else 65
        calls = 0

        def batch(self, *args, **kwargs):
            self.calls += 1
            return {"status": "aborted" if self.calls == 4 else "committed"}

    store = AntflyJSONStore(client=Client(), table="t", batch_size=2)
    with pytest.raises(WriteOutcomeError) as caught:
        if operation == "mset":
            store.mset([(str(i), "x" * 100) for i in range(4)])
        else:
            store.mdelete([str(i) for i in range(4)])
    assert caught.value.committed_keys == ("0", "1", "2")
    assert caught.value.keys == ("3",)


@pytest.mark.asyncio
async def test_cancel_key_iteration_preserves_cancellation_and_closes_after_worker():
    import threading

    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Store(AntflyJSONStore):
        def yield_keys(self, *, prefix=None):
            try:
                started.set()
                release.wait(2)
                yield "a"
            finally:
                closed.set()

    iterator = Store(client=None, table="t").ayield_keys()
    task = asyncio.create_task(anext(iterator))
    while not started.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not closed.is_set()
    finally:
        release.set()
    for _ in range(100):
        if closed.is_set():
            break
        await asyncio.sleep(0.005)
    assert closed.is_set()
