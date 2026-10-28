"""Shared transport, outcome handling, and bounded async execution."""

from __future__ import annotations

import asyncio
import functools
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import quote

_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="antfly-langchain")


class WriteOutcomeError(RuntimeError):
    """A write did not establish requested visibility. Never blindly replay it."""

    def __init__(self, status: str, keys=(), details=None, committed_keys=()):
        self.status = status
        self.keys = tuple(keys)
        self.committed_keys = tuple(committed_keys)
        self.details = details
        self.committed = True if status.startswith("committed") else False if status == "aborted" else None
        super().__init__(
            f"Antfly write outcome {status} for {len(self.keys)} keys: {details}. "
            "Inspect committed data/index status before retrying; mutations are not replayed."
        )


def response_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, dict):
        raise ValueError("Antfly returned a non-object response")
    return value


def request(client, method: str, path: str, **kwargs):
    """Use the public SDK transport, including status codes and byte bounds."""
    return client.request(method, path, **kwargs)


def check_write_result(result, keys=(), wait=True):
    try:
        body = response_dict(result)
    except (ValueError, TypeError, AttributeError) as exc:
        raise WriteOutcomeError("unknown", keys, "Antfly returned an unusable mutation outcome") from exc
    status = body.get("status")
    if not isinstance(status, str):
        raise WriteOutcomeError("unknown", keys, body)
    if status == "committed" or (status == "committed_pending" and not wait):
        return body
    raise WriteOutcomeError(status or "unknown", keys, body.get("failure", body))


def _write_batch(client, table, inserts, deletes, wait, transforms=None):
    keys = list((inserts or {}).keys()) + list(deletes or [])
    keys += [x["key"] for x in (transforms or [])]
    try:
        result = client.batch(
            table,
            inserts=inserts,
            deletes=deletes,
            transforms=transforms,
            sync_level="full_index" if wait else "write",
        )
    except (TimeoutError, OSError) as exc:
        raise WriteOutcomeError("unknown", keys, str(exc)) from exc
    except Exception as exc:
        import httpx
        from antfly import AntflyHTTPError, AntflyResponseError

        if isinstance(exc, (httpx.TransportError, AntflyResponseError)) or (
            isinstance(exc, AntflyHTTPError) and (exc.status_code == 408 or exc.status_code >= 500)
        ):
            raise WriteOutcomeError("unknown", keys, str(exc)) from exc
        raise
    return check_write_result(result, keys, wait)


def write(client, table, inserts=None, deletes=None, wait=True, transforms=None):
    """Bound request size and report partial progress without mutation replay.

    Transforms are one indivisible batch (memory markers and rows are atomic).
    Ordinary insert/delete workloads split at 64 records or the SDK byte limit.
    """
    if transforms is not None:
        return _write_batch(client, table, inserts, deletes, wait, transforms)
    rows = [("insert", key, value) for key, value in (inserts or {}).items()]
    rows += [("delete", key, None) for key in (deletes or [])]
    limit = getattr(client, "max_write_request_bytes", 64 << 20)
    if not isinstance(limit, int) or limit <= 0:
        limit = 64 << 20

    def encoded(value):
        return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8"))

    overhead = encoded({"inserts": {}, "deletes": [], "sync_level": "full_index" if wait else "write"})
    sizes = [encoded({key: value}) if kind == "insert" else encoded(key) + 1 for kind, key, value in rows]
    if any(size + overhead > limit for size in sizes):
        raise ValueError(f"A document exceeds the SDK {limit} byte write limit; no mutations were submitted")
    batches, batch, used = [], [], overhead
    for row, size in zip(rows, sizes, strict=True):
        if batch and (len(batch) >= 64 or used + size > limit):
            batches.append(batch)
            batch, used = [], overhead
        batch.append(row)
        used += size
    if batch:
        batches.append(batch)
    completed_keys, outcome = [], {"status": "committed"}
    for batch in batches:
        keys = [key for _, key, _ in batch]
        try:
            outcome = _write_batch(
                client,
                table,
                {key: value for kind, key, value in batch if kind == "insert"} or None,
                [key for kind, key, _ in batch if kind == "delete"] or None,
                wait,
            )
        except WriteOutcomeError as exc:
            exc.committed_keys = tuple(completed_keys) + exc.committed_keys
            raise
        except Exception as exc:
            if completed_keys:
                raise WriteOutcomeError("partial", keys, str(exc), completed_keys) from exc
            raise
        completed_keys.extend(keys)
    return outcome


def hits(response) -> list[dict[str, Any]]:
    """Read one table query without turning omitted evidence into empty hits."""
    body = response_dict(response)
    responses = body.get("responses", [body])
    if not isinstance(responses, list) or len(responses) != 1 or not isinstance(responses[0], dict):
        raise RuntimeError("Antfly query requires exactly one response object for its table")
    response = responses[0]
    if response.get("status", 200) != 200 or response.get("error"):
        raise RuntimeError(f"Antfly query failed: {response}")
    group = response.get("hits")
    found = group.get("hits") if isinstance(group, dict) else group
    if not isinstance(found, list) or any(not isinstance(hit, dict) for hit in found):
        raise RuntimeError("Antfly query omitted or returned malformed hits")
    return found


async def run_async(fn, *args, **kwargs):
    # Bound queued as well as running operations. One semaphore per event loop
    # avoids retaining closed loops and permits independent asyncio.run calls.
    loop = asyncio.get_running_loop()
    sem = getattr(loop, "_antfly_concurrency", None)
    if sem is None:
        sem = asyncio.Semaphore(16)
        loop._antfly_concurrency = sem
    await sem.acquire()
    try:
        future = loop.run_in_executor(_EXECUTOR, functools.partial(fn, *args, **kwargs))
    except BaseException:
        sem.release()
        raise

    def completed(done):
        sem.release()
        if not done.cancelled():
            done.exception()  # Observe worker errors even if the awaiting task was cancelled.

    future.add_done_callback(completed)
    return await asyncio.shield(future)


def path_part(value: str) -> str:
    return quote(value, safe="")


def json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
