"""Cancellation preserves uncertain mutation semantics and executor boundaries."""

import asyncio
import threading

import httpx
import pytest
from langchain_antfly import AntflyJSONStore
from langchain_antfly._utils import run_async
from test_stores import MemoryClient


@pytest.mark.asyncio
async def test_cancelled_write_can_commit_once_and_lost_response_is_observed():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    class DelayedReply(MemoryClient):
        def batch(self, table, **kwargs):
            super().batch(table, **kwargs)
            entered.set()
            try:
                assert release.wait(3)
                raise httpx.ReadTimeout("response lost after commit")
            finally:
                finished.set()

    client = DelayedReply()
    store = AntflyJSONStore.create(client=client, table="cancelled")
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    unhandled = []
    loop.set_exception_handler(lambda loop, context: unhandled.append(context))
    task = asyncio.create_task(store.amset([("explicit-key", {"committed": True})]))
    try:
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The event loop and reads remain usable while the cancelled worker waits.
        assert store.mget(["explicit-key"]) == [{"committed": True}]
        await asyncio.sleep(0.001)
        assert not finished.is_set()
    finally:
        release.set()
        for _ in range(100):
            if finished.is_set():
                break
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.01)
        loop.set_exception_handler(previous)
    assert finished.is_set()
    assert client.calls == 1
    assert unhandled == []


@pytest.mark.asyncio
async def test_blocking_workers_remain_bounded_and_loop_stays_responsive():
    release = threading.Event()
    lock = threading.Lock()
    active = peak = started = 0

    def blocking():
        nonlocal active, peak, started
        with lock:
            started += 1
            active += 1
            peak = max(peak, active)
        try:
            assert release.wait(3)
        finally:
            with lock:
                active -= 1

    tasks = [asyncio.create_task(run_async(blocking)) for _ in range(40)]
    try:
        for _ in range(100):
            with lock:
                ready = started == 8
            if ready:
                break
            await asyncio.sleep(0.005)
        assert ready
        # All running workers are blocked; a heartbeat still completes promptly.
        await asyncio.wait_for(asyncio.sleep(0.001), timeout=0.25)
        assert peak == 8
        assert loop_slots_in_use() == 16
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert started == 40
    assert peak <= 8


def loop_slots_in_use():
    return 16 - asyncio.get_running_loop()._antfly_concurrency._value
