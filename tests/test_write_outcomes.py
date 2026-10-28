"""Ambiguous mutations are reported without a replay or precommit reclassification."""

import httpx
import pytest
from antfly import (
    AntflyClient,
    AntflyHTTPError,
    AntflyResponseError,
    IndexMutationTemporarilyUnavailableError,
    StorageResourceExhaustedError,
)
from langchain_antfly._utils import WriteOutcomeError, check_write_result, write


@pytest.mark.parametrize("body", [b"{", b"[]", b"null", b"{}", b'{"status":"future_unknown"}'])
def test_real_sdk_malformed_write_reply_remains_unknown(body):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=body)

    client = AntflyClient("http://test")
    client._client.set_httpx_client(httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)))
    try:
        with pytest.raises(WriteOutcomeError) as caught:
            write(client, "t", inserts={"a": {}})
        assert caught.value.status == "unknown"
        assert caught.value.committed is None
        assert caught.value.keys == ("a",)
        assert isinstance(caught.value.__cause__, AntflyResponseError)
        assert len(calls) == 1
    finally:
        client.close()


@pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
def test_ambiguous_http_write_error_is_unknown(status):
    class Client:
        calls = 0

        def batch(self, *a, **kw):
            self.calls += 1
            raise AntflyHTTPError(status, "failed")

    client = Client()
    with pytest.raises(WriteOutcomeError) as caught:
        write(client, "t", inserts={"a": {}})
    assert caught.value.status == "unknown"
    assert client.calls == 1


@pytest.mark.parametrize(
    "error",
    [
        IndexMutationTemporarilyUnavailableError("index_probe_unavailable", "precommit"),
        StorageResourceExhaustedError("precommit", 10),
        AntflyHTTPError(403, "forbidden"),
    ],
)
def test_known_precommit_rejections_remain_specialized(error):
    class Client:
        def batch(self, *a, **kw):
            raise error

    with pytest.raises(type(error)) as caught:
        write(Client(), "t", inserts={"a": {}})
    assert caught.value is error


@pytest.mark.parametrize("value", [None, [], {"status": []}, {"status": 200}])
def test_unusable_custom_client_outcome_is_unknown(value):
    with pytest.raises(WriteOutcomeError) as caught:
        check_write_result(value, ["a"])
    assert caught.value.status == "unknown"


@pytest.mark.parametrize("status,body", [(408, b"timeout"), (500, b"internal"), (503, b"unavailable"), (204, b"")])
def test_real_sdk_uncertain_http_outcomes_are_not_replayed(status, body):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, content=body)

    client = AntflyClient("http://test")
    client._client.set_httpx_client(httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)))
    try:
        with pytest.raises(WriteOutcomeError) as caught:
            write(client, "t", inserts={"a": {}})
        assert caught.value.status == "unknown"
        assert caught.value.committed is None
        assert len(calls) == 1
    finally:
        client.close()


@pytest.mark.parametrize(
    "body",
    [
        {"responses": []},
        {"responses": None},
        {"responses": {}},
        {"responses": [None]},
        {"responses": [{"hits": []}, {"hits": []}]},
        {"responses": [{}]},
        {"hits": None},
        {"hits": {}},
        {"hits": {"hits": None}},
        {"hits": {"hits": {}}},
        {"hits": [None]},
        {"hits": ""},
    ],
)
def test_omitted_or_malformed_query_hits_never_masquerade_as_empty_results(body):
    from langchain_antfly._utils import hits

    with pytest.raises(RuntimeError):
        hits(body)


@pytest.mark.parametrize("body", [{"hits": []}, {"hits": {"hits": []}}, {"responses": [{"hits": {"hits": []}}]}])
def test_explicit_empty_query_hits_are_valid(body):
    from langchain_antfly._utils import hits

    assert hits(body) == []
