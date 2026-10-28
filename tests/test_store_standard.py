"""Unmodified upstream synchronous and asynchronous store suites."""

import os
import uuid

import pytest
from antfly import AntflyClient
from langchain_antfly import AntflyByteStore, AntflyDocumentStore, AntflyJSONStore
from langchain_core.documents import Document
from langchain_tests.integration_tests.base_store import BaseStoreAsyncTests, BaseStoreSyncTests
from test_stores import MemoryClient


class _Fixtures:
    @pytest.fixture
    def kv_store(self):
        url = os.getenv("ANTFLY_TEST_URL")
        client = AntflyClient(url, token=os.getenv("ANTFLY_TEST_TOKEN")) if url else MemoryClient()
        table = "lc_standard_" + uuid.uuid4().hex[:12] if url else "test"
        store = self.store_class.create(client=client, table=table, page_size=1)
        try:
            yield store
        finally:
            if url:
                client.drop_table(table)
                client.close()

    @pytest.fixture
    def three_values(self):
        if self.store_class is AntflyByteStore:
            return b"\x00\xff", b"", b"hello"
        if self.store_class is AntflyDocumentStore:
            return tuple(Document(id=str(i), page_content=str(i), metadata={"p": i}) for i in range(3))
        return {"x": [1, True]}, {}, {"y": "é"}


class TestJSONSync(_Fixtures, BaseStoreSyncTests):
    store_class = AntflyJSONStore


class TestJSONAsync(_Fixtures, BaseStoreAsyncTests):
    store_class = AntflyJSONStore


class TestDocumentSync(_Fixtures, BaseStoreSyncTests):
    store_class = AntflyDocumentStore


class TestDocumentAsync(_Fixtures, BaseStoreAsyncTests):
    store_class = AntflyDocumentStore


class TestBytesSync(_Fixtures, BaseStoreSyncTests):
    store_class = AntflyByteStore


class TestBytesAsync(_Fixtures, BaseStoreAsyncTests):
    store_class = AntflyByteStore
