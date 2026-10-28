"""Parse our declared fixture, keeping actual images separate from proxies."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from .evidence import sha256

FIXTURES = Path(__file__).parent / "fixtures"
PDF = FIXTURES / "orchard-report.pdf"


def extract(pdf: Path = PDF) -> tuple[list[Document], list[Document], dict[str, bytes]]:
    """Fixture-specific extraction; not a universal arbitrary-PDF parser."""
    manifest = json.loads((FIXTURES / "answers.json").read_text())
    source_hash = sha256(pdf.read_bytes())
    prefix = source_hash[:16]
    parents = []
    children = []
    media = {}
    with pymupdf.open(pdf) as document:
        if len(document) != 3:
            raise ValueError("Expected the declared three-page fixture")
        for page_number, page in enumerate(document, 1):
            if page_number == 1:
                source_id = "page-1"
                body = page.get_text().strip()
                metadata = {"source_id": source_id, "page": page_number, "source_sha256": source_hash}
                proxy = body
            else:
                source_id = f"figure-{page_number - 1}"
                images = page.get_images(full=True)
                if len(images) != 1:
                    raise ValueError(f"Expected one embedded figure on page {page_number}, got {len(images)}")
                payload = document.extract_image(images[0][0])
                value = payload["image"]
                key = f"{prefix}:{source_id}:media"
                media[key] = value
                metadata = {
                    "source_id": source_id,
                    "page": page_number,
                    "source_sha256": source_hash,
                    "media_key": key,
                    "media_sha256": sha256(value),
                    "mime_type": f"image/{payload['ext']}",
                }
                body = page.get_text().strip()
                proxy = manifest["summaries"][source_id]
            key = f"{prefix}:{source_id}"
            parents.append(Document(id=key, page_content=body, metadata=metadata))
            children.append(
                Document(
                    id=f"{key}:{manifest['summary_revision']}",
                    page_content=proxy,
                    metadata={"doc_id": key, "source_id": source_id, "source_sha256": source_hash},
                )
            )
    return parents, children, media
