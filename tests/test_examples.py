"""Deterministic evidence-boundary tests; these are not live-model evaluations."""

import base64
import json

import pytest
from langchain_core.documents import Document

from examples.multimodal.corpus import FIXTURES, extract
from examples.multimodal.evidence import IncompleteEvidenceError, aresolve, resolve, sha256


class Store:
    def __init__(self, values):
        self.values = dict(values)

    def mget(self, keys):
        return [self.values.get(key) for key in keys]

    async def amget(self, keys):
        return self.mget(keys)


def stores():
    parents, children, media = extract()
    return children, Store((parent.id, parent) for parent in parents), Store(media)


def test_fixture_has_portable_visual_evidence_and_stable_ids():
    parents, children, media = extract()
    repeated = extract()
    assert [p.id for p in parents] == [p.id for p in repeated[0]]
    assert len(parents) == len(children) == 3
    assert len(media) == 2
    assert "Mira Chen" in parents[0].page_content
    visual = json.loads((FIXTURES / "answers.json").read_text())
    assert "orange" not in visual["summaries"]["figure-1"].lower()
    assert "circle" not in visual["summaries"]["figure-1"].lower()
    assert all("/" not in p.metadata.get("media_key", "") for p in parents)
    assert len(visual["questions"]) == 20
    assert {item["category"] for item in visual["questions"]} == {"text", "table", "visual", "cross_modal", "negative"}


def test_exact_child_set_resolves_real_images_into_prompt():
    children, parents, media = stores()
    evidence = resolve([children[1]], parents, media)
    parts, trace = evidence.content_parts()
    assert [p.metadata["source_id"] for p in evidence.parents] == ["figure-1"]
    image = next(p for p in parts if p["type"] == "image_url")
    prefix, payload = image["image_url"]["url"].split(",", 1)
    value = base64.b64decode(payload)
    assert prefix == "data:image/png;base64"
    assert sha256(value) == trace[0]["sha256"]
    assert value == media.values[trace[0]["media_key"]]


@pytest.mark.parametrize("failure", ["missing_id", "missing_parent", "missing_media", "corrupt_media"])
def test_broken_evidence_prevents_generation(failure):
    children, parents, media = stores()
    children = [children[1]]
    if failure == "missing_id":
        children = [Document(page_content="bad reference")]
    elif failure == "missing_parent":
        parents.values.clear()
    elif failure == "missing_media":
        media.values.clear()
    else:
        media.values = dict.fromkeys(media.values, b"corrupted")
    with pytest.raises(IncompleteEvidenceError):
        resolve(children, parents, media)


async def test_async_evidence_has_same_strict_semantics():
    children, parents, media = stores()
    assert (await aresolve(children, parents, media)).content_parts() == resolve(
        children, parents, media
    ).content_parts()
    parents.values.clear()
    with pytest.raises(IncompleteEvidenceError, match="Missing parent"):
        await aresolve(children, parents, media)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_answer_boundary_never_calls_model_with_missing_media(monkeypatch, asynchronous):
    from examples.multimodal import __main__ as app

    children, parents, media = stores()
    media.values.clear()
    calls = []

    class Vectors:
        def similarity_search(self, query, k):
            return [children[1]]

        async def asimilarity_search(self, query, k):
            return [children[1]]

    monkeypatch.setattr(app, "attach", lambda *_: (Vectors(), parents, media))
    monkeypatch.setattr(app, "_vision", lambda *_: calls.append("generation"))
    with pytest.raises(IncompleteEvidenceError, match="Missing media"):
        if asynchronous:
            await app.aask(None, "test", "model", "question", live=True)
        else:
            app.ask(None, "test", "model", "question", live=True)
    assert calls == []


@pytest.mark.parametrize("protocol", ["openai", "gemini"])
def test_provider_request_preserves_actual_image_bytes(monkeypatch, protocol):
    from examples.multimodal import __main__ as app

    children, parents, media = stores()
    parts, trace = resolve([children[1]], parents, media).content_parts()
    monkeypatch.setenv("VISION_PROTOCOL", protocol)
    monkeypatch.setenv("VISION_URL", "https://example.invalid/generate")
    monkeypatch.setenv("VISION_MODEL", "test-model")
    monkeypatch.setenv("VISION_API_KEY", "unit-test-placeholder")
    captured = []

    class Response:
        is_error = False

        def json(self):
            if protocol == "openai":
                return {"choices": [{"message": {"content": "orange circle [figure-1]"}}]}
            return {"candidates": [{"content": {"parts": [{"text": "orange circle [figure-1]"}]}}]}

    def post(url, **kwargs):
        captured.append(kwargs["json"])
        return Response()

    monkeypatch.setattr(app.httpx, "post", post)
    assert app._vision(parts, "Describe marker") == "orange circle [figure-1]"
    if protocol == "openai":
        image = next(p for p in captured[0]["messages"][1]["content"] if p["type"] == "image_url")
        encoded = image["image_url"]["url"].split(",", 1)[1]
    else:
        image = next(p for p in captured[0]["contents"][0]["parts"] if "inlineData" in p)
        encoded = image["inlineData"]["data"]
    assert sha256(base64.b64decode(encoded)) == trace[0]["sha256"]


class IngestionBackend:
    """Fail publication stages independently while keeping durable checkpoints."""

    def __init__(self):
        self.tables = {}
        self.vector_rows = {}
        self.embedding_rows = {}
        self.fail = None
        self.embedding_calls = 0

    def list_tables(self):
        return [{"name": name} for name in self.tables]

    def get(self, table, key, fields=None):
        return {"_embeddings": {"vectors": self.embedding_rows[key]}}

    def classes(self):
        backend = self

        class PersistentStore(Store):
            max_value_bytes = 8 * 1024 * 1024

            def __init__(self, *, client, table):
                self.values = backend.tables[table]
                self.table = table

            @classmethod
            def create(cls, *, client, table):
                backend.tables[table] = {}
                return cls(client=client, table=table)

            def mset(self, pairs):
                for key, value in pairs:
                    stage = "media" if self.table.endswith("_media") else "parents"
                    if key.startswith("transform-v1:") or key == "ingestion-pending":
                        stage = "checkpoint"
                    if backend.fail == stage:
                        backend.fail = None
                        raise RuntimeError(f"Interrupted {stage}")
                    self.values[key] = value

            def mdelete(self, keys):
                for key in keys:
                    self.values.pop(key, None)

        class Provider:
            def __init__(self, **kwargs):
                pass

            def embed_documents(self, texts):
                backend.embedding_calls += len(texts)
                return [[1.0, 0.5] for text in texts]

        class Vectors:
            def __init__(self, *, client, table, embedding, **kwargs):
                self.table = table

            @classmethod
            def from_documents(cls, children, *, embedding, client, table, **kwargs):
                dense = embedding.embed_documents([c.page_content for c in children])
                if backend.fail == "vectors":
                    backend.fail = None
                    raise RuntimeError("Interrupted vectors")
                backend.tables.setdefault(table, {})
                backend.vector_rows.update({c.id: c for c in children})
                backend.embedding_rows.update({c.id: value for c, value in zip(children, dense, strict=True)})
                return cls(client=client, table=table, embedding=embedding)

            def get_by_ids(self, ids):
                return [backend.vector_rows[key] for key in ids if key in backend.vector_rows]

            def delete(self, ids):
                for key in ids:
                    backend.vector_rows.pop(key, None)
                    backend.embedding_rows.pop(key, None)

        return PersistentStore, PersistentStore, Provider, Vectors


def ingestion_setup(monkeypatch):
    import langchain_classic.retrievers.multi_vector as multi

    from examples.multimodal import __main__ as app

    backend = IngestionBackend()
    monkeypatch.setattr(app, "_classes", backend.classes)
    monkeypatch.setattr(multi, "MultiVectorRetriever", lambda **kwargs: None)
    return app, backend


@pytest.mark.parametrize("stage", ["media", "parents", "vectors"])
def test_ingestion_resumes_stage_failures_without_repeating_completed_transforms(monkeypatch, stage):
    app, backend = ingestion_setup(monkeypatch)
    calls = []
    monkeypatch.setattr(app, "_vision", lambda *args: calls.append(1) or "Shipments chart and system components")
    backend.fail = stage
    with pytest.raises(RuntimeError, match="Interrupted"):
        app.ingest(backend, "test", "model", regenerate=True)
    calls_before_retry = len(calls)
    embedding_before_retry = backend.embedding_calls
    result = app.ingest(backend, "test", "model", regenerate=True)
    assert result["status"] == "ingested"
    assert len(calls) == calls_before_retry == 2
    if stage == "vectors":
        assert backend.embedding_calls == embedding_before_retry
    assert app.ingest(backend, "test", "model")["status"] == "unchanged"


@pytest.mark.parametrize("missing", ["media", "parent", "vector"])
def test_completed_ingestion_repairs_missing_records_without_model_calls(monkeypatch, missing):
    app, backend = ingestion_setup(monkeypatch)
    app.ingest(backend, "test", "model")
    manifest = backend.tables["test_parents"]["ingestion-manifest"]
    if missing == "media":
        del backend.tables["test_media"][manifest.metadata["media"][0]]
    elif missing == "parent":
        del backend.tables["test_parents"][manifest.metadata["parents"][0]]
    else:
        del backend.vector_rows[manifest.metadata["children"][0]]
    result = app.ingest(backend, "test", "model")
    assert result["status"] == "ingested"
    assert result["model_calls"] == 0
    assert backend.embedding_calls == 3
    assert app.ingest(backend, "test", "model")["status"] == "unchanged"


def test_changed_source_replaces_owned_references_after_interrupted_publication(monkeypatch):
    app, backend = ingestion_setup(monkeypatch)
    app.ingest(backend, "test", "model")
    old = set(backend.vector_rows)
    original_extract = app.extract

    def changed():
        parents, children, media = original_extract()
        for parent, child in zip(parents, children, strict=True):
            parent.id += ":changed"
            child.id += ":changed"
            child.metadata["doc_id"] = parent.id
            child.page_content += " revised"
        return parents, children, media

    monkeypatch.setattr(app, "extract", changed)
    # Transform revision is part of the fingerprint, as when fixture summaries change.
    monkeypatch.setattr(app, "sha256", lambda value: "changed-" + sha256(value))
    backend.fail = "vectors"
    with pytest.raises(RuntimeError, match="Interrupted vectors"):
        app.ingest(backend, "test", "model")
    assert old.issubset(backend.vector_rows)
    count = backend.embedding_calls
    app.ingest(backend, "test", "model")
    assert backend.embedding_calls == count
    assert not old.intersection(backend.vector_rows)


def test_partial_caption_resume_retains_live_mode_without_repeating_completed_caption(monkeypatch):
    app, backend = ingestion_setup(monkeypatch)
    calls = []

    def caption(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("Caption provider failed")
        return "A searchable figure summary"

    monkeypatch.setattr(app, "_vision", caption)
    with pytest.raises(RuntimeError, match="Caption provider"):
        app.ingest(backend, "test", "model", regenerate=True)
    result = app.ingest(backend, "test", "model")
    assert result["captions"] == "live"
    assert len(calls) == 3  # One completed caption reused; only failed item called again.
    assert "ingestion-pending" not in backend.tables["test_parents"]


def preflight_client(monkeypatch):
    from types import SimpleNamespace

    from examples.multimodal import __main__ as app

    class Client:
        base_url = "http://localhost:8080"

        def models(self):
            return {"embedders": {"model": {"inputs": ["text"], "inference_capabilities": {"task": "embed"}}}}

        def list_tables(self):
            return [{"name": "test_vectors"}]

    checks = []

    class Vectors:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.dimension = None

        def validate(self):
            checks.append(self.kwargs)

    monkeypatch.setattr(app, "_classes", lambda: (None, None, lambda **kwargs: None, Vectors))
    monkeypatch.setattr(app, "attach", lambda *args: (None, None, SimpleNamespace(max_value_bytes=8 * 1024 * 1024)))
    for name in ("VISION_URL", "VISION_MODEL", "VISION_API_KEY", "VISION_PROTOCOL"):
        monkeypatch.delenv(name, raising=False)
    return app, Client(), checks


def test_preflight_validates_existing_index_without_inference(monkeypatch):
    app, client, checks = preflight_client(monkeypatch)
    result = app.preflight(client, "test", "model")
    assert result["model_calls"] == 0
    assert checks[0]["distance_metric"] == "cosine"


@pytest.mark.parametrize("failure", ["index", "media_limit", "vision_credentials", "vision_pair", "modality"])
def test_preflight_rejects_incompatible_configuration_without_model_calls(monkeypatch, failure):
    from types import SimpleNamespace

    app, client, _ = preflight_client(monkeypatch)
    if failure == "index":

        class InvalidVectors:
            def __init__(self, **kwargs):
                pass

            def validate(self):
                raise ValueError("Existing index incompatible")

        monkeypatch.setattr(app, "_classes", lambda: (None, None, lambda **kwargs: None, InvalidVectors))
    elif failure == "media_limit":
        monkeypatch.setattr(app, "attach", lambda *args: (None, None, SimpleNamespace(max_value_bytes=1)))
    elif failure == "vision_credentials":
        monkeypatch.setenv("VISION_URL", "https://provider.example/v1/chat/completions")
        monkeypatch.setenv("VISION_MODEL", "vision")
    elif failure == "vision_pair":
        monkeypatch.setenv("VISION_MODEL", "vision")
    else:
        monkeypatch.setattr(client, "models", lambda: {"embedders": {"model": {"inputs": ["image"]}}})
    with pytest.raises(ValueError):
        app.preflight(client, "test", "model")


@pytest.mark.parametrize(
    "corruption",
    ["child_content", "child_parent_reference", "parent_content", "parent_provenance", "dense_vector", "media_bytes"],
)
def test_completed_ingestion_repairs_corrupt_evidence_without_model_calls(monkeypatch, corruption):
    app, backend = ingestion_setup(monkeypatch)
    app.ingest(backend, "test", "model")
    expected_parents, expected_children, expected_media = app.extract()
    child = backend.vector_rows[expected_children[0].id]
    if corruption == "child_content":
        child.page_content = "Corrupt searchable content"
    elif corruption == "child_parent_reference":
        child.metadata["doc_id"] = "nonexistent-parent"
    elif corruption == "dense_vector":
        backend.embedding_rows[expected_children[0].id] = [0.0, 1.0]
    elif corruption == "parent_content":
        backend.tables["test_parents"][expected_parents[0].id].page_content = "Corrupt original evidence"
    elif corruption == "parent_provenance":
        backend.tables["test_parents"][expected_parents[0].id].metadata["source_sha256"] = "wrong-source"
    else:
        backend.tables["test_media"][next(iter(expected_media))] = b"corrupt media bytes"
    calls = backend.embedding_calls
    result = app.ingest(backend, "test", "model")
    assert result["status"] == "ingested"
    assert result["model_calls"] == 0
    assert backend.embedding_calls == calls
    for expected in expected_children:
        actual = backend.vector_rows[expected.id]
        assert actual.page_content == expected.page_content
        assert actual.metadata == expected.metadata
        assert backend.embedding_rows[expected.id] == [1.0, 0.5]
    for expected in expected_parents:
        actual = backend.tables["test_parents"][expected.id]
        assert actual.page_content == expected.page_content
        assert actual.metadata == expected.metadata
    for key, value in expected_media.items():
        assert backend.tables["test_media"][key] == value
    assert app.ingest(backend, "test", "model")["status"] == "unchanged"


@pytest.mark.parametrize("revert", [False, True])
def test_source_change_after_partial_publication_cleans_every_abandoned_revision(monkeypatch, revert):
    app, backend = ingestion_setup(monkeypatch)
    app.ingest(backend, "test", "model")
    original_extract, original_resolve = app.extract, app.resolve
    revision = "B"

    def changed():
        parents, children, media = original_extract()
        renamed_media = {key + ":" + revision: value for key, value in media.items()}
        for parent, child in zip(parents, children, strict=True):
            parent.id += ":" + revision
            child.id += ":" + revision
            child.metadata["doc_id"] = parent.id
            child.page_content += " " + revision
            if parent.metadata.get("media_key"):
                parent.metadata["media_key"] += ":" + revision
        return parents, children, renamed_media

    monkeypatch.setattr(app, "extract", changed)
    monkeypatch.setattr(app, "sha256", lambda value: revision + "-" + sha256(value))

    def interrupted(*args):
        raise RuntimeError("Interrupted after vector publication")

    monkeypatch.setattr(app, "resolve", interrupted)
    with pytest.raises(RuntimeError, match="after vector publication"):
        app.ingest(backend, "test", "model")
    assert any(key.endswith(":B") for key in backend.vector_rows)
    assert any(key.endswith(":B") for key in backend.tables["test_parents"])
    assert any(key.endswith(":B") for key in backend.tables["test_media"])
    revision = "C"
    monkeypatch.setattr(app, "resolve", original_resolve)
    if revert:
        monkeypatch.setattr(app, "extract", original_extract)
        monkeypatch.setattr(app, "sha256", sha256)
    app.ingest(backend, "test", "model")
    expected_parents, expected_children, expected_media = app.extract()
    assert set(backend.vector_rows) == {child.id for child in expected_children}
    actual_parents = {key for key in backend.tables["test_parents"] if not key.startswith("transform-v1:")}
    assert actual_parents == {parent.id for parent in expected_parents} | {"ingestion-manifest"}
    assert set(backend.tables["test_media"]) == set(expected_media)
    assert app.ingest(backend, "test", "model")["status"] == "unchanged"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["swapped_existing_parent", "wrong_source_hash"])
async def test_conflicting_child_provenance_blocks_generation(monkeypatch, asynchronous, failure):
    from examples.multimodal import __main__ as app

    children, parents, media = stores()
    child = children[1]
    if failure == "swapped_existing_parent":
        child.metadata["doc_id"] = children[2].metadata["doc_id"]
    else:
        child.metadata["source_sha256"] = "different-source-hash"
    calls = []

    class Vectors:
        def similarity_search(self, query, k):
            return [child]

        async def asimilarity_search(self, query, k):
            return [child]

    monkeypatch.setattr(app, "attach", lambda *_: (Vectors(), parents, media))
    monkeypatch.setattr(app, "_vision", lambda *_: calls.append("generation"))
    with pytest.raises(IncompleteEvidenceError, match="disagrees with parent"):
        if asynchronous:
            await app.aask(None, "test", "model", "question", live=True)
        else:
            app.ask(None, "test", "model", "question", live=True)
    assert calls == []


async def test_generic_child_without_optional_source_provenance_remains_supported():
    children, parents, media = stores()
    child = Document(page_content="generic proxy", metadata={"doc_id": children[1].metadata["doc_id"]})
    assert resolve([child], parents, media).parents
    assert (await aresolve([child], parents, media)).parents
