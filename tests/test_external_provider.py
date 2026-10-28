"""Explicit optional live-provider gate; configured prerequisites must fail loudly."""

import os
import uuid

import pytest
from antfly import AntflyClient

from examples import external_embeddings


@pytest.mark.asyncio
async def test_live_unchanged_openai_provider_sync_async_factories_and_retrieval(monkeypatch):
    provider = os.getenv("ANTFLY_TEST_EXTERNAL_PROVIDER")
    if not provider:
        pytest.skip("Set ANTFLY_TEST_EXTERNAL_PROVIDER=openai to run optional third-party provider validation")
    assert provider == "openai", "Unsupported configured ANTFLY_TEST_EXTERNAL_PROVIDER"
    url, model = os.getenv("ANTFLY_TEST_URL"), os.getenv("ANTFLY_TEST_EMBEDDING_MODEL")
    assert url and model, "Configured external provider gate requires ANTFLY_TEST_URL and ANTFLY_TEST_EMBEDDING_MODEL"
    # This import must fail (not skip) when an explicitly requested package is absent.
    try:
        import langchain_openai  # noqa: F401
    except ImportError:
        pytest.fail("Configured external provider requires: uv sync --group interoperability")
    monkeypatch.setenv("ANTFLY_TOKEN", os.getenv("ANTFLY_TEST_TOKEN", ""))
    prefix = "audit_interface_fix_openai_" + uuid.uuid4().hex[:10]
    with AntflyClient(url, token=os.getenv("ANTFLY_TEST_TOKEN"), timeout=120) as client:
        report = await external_embeddings.run(client, model=model, table_prefix=prefix)
        assert report["passed"]
        assert report["sync_document_ids"] == report["async_document_ids"] == ["owner"]
        assert report["cloud_calls"] == 0
        assert len(report["removed_owned_tables"]) == 2
        assert not {item.get("name", item.get("table_name")) for item in client.list_tables()}.intersection(
            report["removed_owned_tables"]
        )


@pytest.mark.asyncio
async def test_optional_provider_missing_is_an_actionable_failure(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_optional(name, *args, **kwargs):
        if name == "langchain_openai":
            raise ModuleNotFoundError("optional package deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_optional)
    with pytest.raises(RuntimeError, match="uv sync --group interoperability"):
        await external_embeddings.run(None, model="explicit", table_prefix="unused")


def test_example_help_does_not_import_optional_provider(monkeypatch, capsys):
    import builtins
    import sys

    original = builtins.__import__

    def without_optional(name, *args, **kwargs):
        if name == "langchain_openai":
            pytest.fail("CLI help must not require optional provider")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_optional)
    monkeypatch.setattr(sys, "argv", ["external_embeddings", "--help"])
    with pytest.raises(SystemExit) as caught:
        external_embeddings.main()
    assert caught.value.code == 0
    assert "--model" in capsys.readouterr().out
