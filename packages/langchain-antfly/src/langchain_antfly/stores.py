"""Persistent LangChain key-value stores with versioned, non-pickle codecs."""

from __future__ import annotations

import asyncio
import base64
import threading
from collections.abc import Iterator, Sequence
from typing import Any

from antfly import AntflyNotFoundError
from langchain_core.documents import Document
from langchain_core.stores import BaseStore

from ._utils import WriteOutcomeError, hits, json_copy, response_dict, run_async, write


class AntflyJSONStore(BaseStore[str, Any]):
    """Store JSON-compatible values; namespace and opaque keys cannot collide.

    Constructors attach; :meth:`create` explicitly creates a new table. Writes
    wait for indexing and report partial progress rather than replaying a batch.
    """

    kind = "json"

    def __init__(self, *, client, table: str, namespace: str = "", page_size: int = 100, batch_size: int = 64):
        if not isinstance(namespace, str):
            raise TypeError("namespace must be a string")
        if page_size < 1 or batch_size < 1:
            raise ValueError("page_size and batch_size must be positive")
        self.client, self.table, self.namespace = client, table, namespace
        self.page_size, self.batch_size = page_size, batch_size
        self._prefix = "lc/" + namespace.encode("utf-8").hex() + "/"

    @classmethod
    def create(cls, *, client, table: str, namespace: str = "", **kwargs):
        client.create_table(table)
        return cls(client=client, table=table, namespace=namespace, **kwargs)

    def _key(self, key: str) -> str:
        if not isinstance(key, str):
            raise TypeError("keys must be strings")
        return self._prefix + key.encode("utf-8").hex()

    def _encode(self, value):
        return json_copy(value)

    def _decode(self, value):
        return json_copy(value)

    def mget(self, keys: Sequence[str]) -> list[Any | None]:
        result = []
        for key in keys:
            try:
                row = self.client.get(self.table, self._key(key))
            except AntflyNotFoundError:
                # A missing table is configuration failure, not a missing value.
                self.client.get_table(self.table)
                result.append(None)
                continue
            if row.get("codec") != {"version": 1, "kind": self.kind} or row.get("key") != key:
                raise ValueError(f"Incompatible {self.kind} codec for key {key!r}")
            result.append(self._decode(row["value"]))
        return result

    def _batches(self, rows):
        for offset in range(0, len(rows), self.batch_size):
            yield rows[offset : offset + self.batch_size]

    def mset(self, key_value_pairs: Sequence[tuple[str, Any]]) -> None:
        # Validate every value before mutating any batch.
        encoded = [
            (key, {"codec": {"version": 1, "kind": self.kind}, "key": key, "value": self._encode(value)})
            for key, value in key_value_pairs
        ]
        for key, _ in encoded:
            self._key(key)
        done = []
        for batch in self._batches(encoded):
            try:
                write(self.client, self.table, inserts={self._key(k): v for k, v in batch})
            except Exception as exc:
                self._raise_outcome(exc, batch, done)
            done.extend(k for k, _ in batch)

    def mdelete(self, keys: Sequence[str]) -> None:
        encoded = [(key, self._key(key)) for key in keys]
        done = []
        for batch in self._batches(encoded):
            try:
                write(self.client, self.table, deletes=[v for _, v in batch])
            except Exception as exc:
                self._raise_outcome(exc, batch, done)
            done.extend(k for k, _ in batch)

    def _raise_outcome(self, exc, batch, done):
        if isinstance(exc, WriteOutcomeError):
            logical = {self._key(key): key for key, _ in batch}
            raise WriteOutcomeError(
                exc.status,
                [logical[key] for key in exc.keys],
                exc.details,
                committed_keys=[*done, *(logical[key] for key in exc.committed_keys)],
            ) from exc
        if done:
            raise WriteOutcomeError("partial", [key for key, _ in batch], str(exc), committed_keys=done) from exc
        raise exc

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        encoded_prefix = self._key(prefix or "")
        cursor = None
        snapshot = None
        while True:
            response = response_dict(
                self.client.query(
                    self.table,
                    filter_prefix=encoded_prefix,
                    limit=self.page_size,
                    order_by=[{"field": "_id", "desc": False}],
                    search_after=cursor,
                    remote_snapshot=snapshot,
                )
            )
            page = hits(response)
            snapshot = response.get("responses", [{}])[0].get("remote_snapshot", snapshot)
            if not page:
                return
            for hit in page:
                key = bytes.fromhex(hit["_id"][len(self._prefix) :]).decode("utf-8")
                yield key
            next_cursor = page[-1].get("_sort")
            if next_cursor is None or next_cursor == cursor:
                if len(page) >= self.page_size:
                    raise RuntimeError("Antfly omitted/failed to advance key cursor; refusing truncated enumeration")
                return
            cursor = next_cursor

    async def amget(self, keys):
        return await run_async(self.mget, keys)

    async def amset(self, pairs):
        await run_async(self.mset, pairs)

    async def amdelete(self, keys):
        await run_async(self.mdelete, keys)

    async def ayield_keys(self, *, prefix=None):
        iterator = self.yield_keys(prefix=prefix)
        sentinel = object()
        pending = None
        stopped = threading.Event()

        def advance():
            try:
                return next(iterator, sentinel)
            finally:
                if stopped.is_set():
                    iterator.close()

        try:
            while True:
                pending = asyncio.create_task(run_async(advance))
                key = await asyncio.shield(pending)
                if key is sentinel:
                    break
                yield key
        finally:
            stopped.set()
            if pending is not None and not pending.done():
                # The worker still owns the generator after cancellation. Close only
                # when next() has finished; observe failures even without an awaiter.
                def close_when_done(task):
                    if not task.cancelled():
                        task.exception()
                        iterator.close()

                pending.add_done_callback(close_when_done)
            elif pending is None or not pending.cancelled():
                iterator.close()


class AntflyDocumentStore(AntflyJSONStore):
    """Persist page_content, metadata and ID without serializing Python objects."""

    kind = "document"

    def _encode(self, value):
        if not isinstance(value, Document):
            raise TypeError("AntflyDocumentStore values must be Document objects")
        return json_copy({"id": value.id, "page_content": value.page_content, "metadata": value.metadata})

    def _decode(self, value):
        return Document(**json_copy(value))


class AntflyByteStore(AntflyJSONStore):
    """Persist opaque bytes using a bounded, lossless base64 codec."""

    kind = "bytes"

    def __init__(self, *, max_value_bytes: int = 8 * 1024 * 1024, **kwargs):
        super().__init__(**kwargs)
        if max_value_bytes < 1:
            raise ValueError("max_value_bytes must be positive")
        self.max_value_bytes = max_value_bytes

    def _encode(self, value):
        if not isinstance(value, bytes):
            raise TypeError("AntflyByteStore values must be bytes")
        if len(value) > self.max_value_bytes:
            raise ValueError(f"Media exceeds {self.max_value_bytes} byte store limit")
        return base64.b64encode(value).decode("ascii")

    def _decode(self, value):
        if not isinstance(value, str) or len(value) > 4 * ((self.max_value_bytes + 2) // 3):
            raise ValueError("Stored media exceeds byte store limit")
        decoded = base64.b64decode(value, validate=True)
        if len(decoded) > self.max_value_bytes:
            raise ValueError("Stored media exceeds byte store limit")
        return decoded
