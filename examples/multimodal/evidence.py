"""Strict, testable evidence boundary between retrieval and generation."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


class IncompleteEvidenceError(ValueError):
    """A retrieved reference cannot be resolved; generation must not proceed."""


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass
class Evidence:
    parents: list[Document]
    media: dict[str, bytes]
    children: list[Document]

    def content_parts(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        parts: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        for parent in self.parents:
            source = parent.metadata["source_id"]
            parts.append({"type": "text", "text": f"[{source}] {parent.page_content}"})
            key = parent.metadata.get("media_key")
            if key:
                value = self.media[key]
                mime = parent.metadata["mime_type"]
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{base64.b64encode(value).decode()}"},
                    }
                )
                trace.append(
                    {
                        "source_id": source,
                        "media_key": key,
                        "sha256": sha256(value),
                        "bytes": len(value),
                        "mime_type": mime,
                    }
                )
        return parts, trace


def _parent_ids(children: list[Document]) -> list[str]:
    result = []
    for child in children:
        key = child.metadata.get("doc_id")
        if not isinstance(key, str) or not key:
            raise IncompleteEvidenceError(f"Retrieved child {child.id!r} has no nonempty metadata.doc_id")
        if key not in result:
            result.append(key)
    return result


def _validate_parents(ids: list[str], values: list[Document | None]) -> list[Document]:
    if len(values) != len(ids):
        raise IncompleteEvidenceError("Parent store returned the wrong number of entries")
    for key, value in zip(ids, values, strict=True):
        if value is None:
            raise IncompleteEvidenceError(f"Missing parent {key!r}; repair ingestion before answering")
        if not isinstance(value.metadata.get("source_id"), str) or not value.metadata["source_id"]:
            raise IncompleteEvidenceError(f"Parent {key!r} lacks source provenance")
    return values  # type: ignore[return-value]


def _validate_child_provenance(children: list[Document], ids: list[str], parents: list[Document]) -> None:
    by_id = dict(zip(ids, parents, strict=True))
    for child in children:
        key = child.metadata["doc_id"]
        for field in ("source_id", "source_sha256"):
            if field in child.metadata:
                value = child.metadata[field]
                if not isinstance(value, str) or not value or value != by_id[key].metadata.get(field):
                    raise IncompleteEvidenceError(
                        f"Retrieved child {child.id!r} {field} disagrees with parent {key!r}; repair ingestion before answering"
                    )


def _validate_media(parents: list[Document], keys: list[str], values: list[bytes | None]) -> dict[str, bytes]:
    if len(values) != len(keys):
        raise IncompleteEvidenceError("Media store returned the wrong number of entries")
    media = {}
    for key, value in zip(keys, values, strict=True):
        if value is None:
            raise IncompleteEvidenceError(f"Missing media {key!r}; generation was prevented")
        media[key] = value
    for parent in parents:
        key = parent.metadata.get("media_key")
        if key:
            if parent.metadata.get("mime_type") not in {"image/png", "image/jpeg"}:
                raise IncompleteEvidenceError(f"Unsupported/missing MIME type for {key!r}")
            if sha256(media[key]) != parent.metadata.get("media_sha256"):
                raise IncompleteEvidenceError(f"Media hash mismatch for {key!r}")
    return media


def resolve(children: list[Document], parents: Any, media_store: Any) -> Evidence:
    """Resolve exactly one captured child result set, without rerunning search."""
    ids = _parent_ids(children)
    documents = _validate_parents(ids, parents.mget(ids))
    _validate_child_provenance(children, ids, documents)
    keys = list(dict.fromkeys(p.metadata["media_key"] for p in documents if p.metadata.get("media_key")))
    media = _validate_media(documents, keys, media_store.mget(keys))
    return Evidence(documents, media, children)


async def aresolve(children: list[Document], parents: Any, media_store: Any) -> Evidence:
    ids = _parent_ids(children)
    documents = _validate_parents(ids, await parents.amget(ids))
    _validate_child_provenance(children, ids, documents)
    keys = list(dict.fromkeys(p.metadata["media_key"] for p in documents if p.metadata.get("media_key")))
    media = _validate_media(documents, keys, await media_store.amget(keys))
    return Evidence(documents, media, children)


async def assert_async_resolution(children: list[Document], parents: Any, media_store: Any) -> Evidence:
    """Convenience for explicit async recipe/testing."""
    return await asyncio.wait_for(aresolve(children, parents, media_store), timeout=30)
