"""Dense text embeddings and explicit image extensions through Antfly inference."""

from __future__ import annotations

import base64
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from ._utils import response_dict, run_async


def validate_vectors(values: Any, count: int, dimension: int | None = None) -> list[list[float]]:
    """Validate a dense provider response without requiring provider metadata."""
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"Expected {count} dense embeddings, received an invalid item count")
    result = []
    for value in values:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("Dense embeddings must be nonempty numeric vectors; sparse embeddings are unsupported")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
            raise ValueError("Embedding values must be finite numbers")
        if dimension is None:
            dimension = len(value)
        if len(value) != dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {dimension}, received {len(value)}")
        if not any(value):
            raise ValueError("Zero vectors are not valid retrieval embeddings")
        result.append([float(x) for x in value])
    return result


def validate_model_info(client: Any, model: str, dimensions: int | None, modality: str) -> None:
    """Reject incompatible advertised facts without loading model weights.

    Discovery does not advertise supported retrieval task types or asset identity;
    the inference request/server remains authoritative for those checks.
    """
    info = response_dict(client.models()).get("embedders", {}).get(model)
    if not isinstance(info, dict):
        raise ValueError(f"Model {model!r} is not an advertised embedding model")
    if (modality == "image" or "inputs" in info) and modality not in info.get("inputs", []):
        raise ValueError(f"Model {model!r} does not advertise {modality} embedding support")
    task = info.get("inference_capabilities", {}).get("task")
    if task is not None and task not in {"embed", "embedding"}:
        raise ValueError(f"Model {model!r} advertises incompatible inference task {task!r}")
    width = info.get("dense_output")
    if dimensions is not None and isinstance(width, dict):
        if not width["minimum_dimensions"] <= dimensions <= width["native_dimensions"]:
            raise ValueError(f"Embedding dimension {dimensions} is outside the model's advertised dense range")


def image_part(image: bytes | str | Path, mime_type: str = "image/png") -> dict[str, Any]:
    if not mime_type.startswith("image/"):
        raise ValueError("mime_type must be an image MIME type")
    if isinstance(image, Path) or (isinstance(image, str) and not image.startswith(("https://", "http://", "data:"))):
        image = Path(image).read_bytes()
    if isinstance(image, bytes):
        if not image:
            raise ValueError("Image bytes must not be empty")
        url = f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"
    else:
        url = image
    return {"type": "image_url", "image_url": {"url": url}}


class AntflyEmbeddings(Embeddings):
    """Use a named Antfly model; construction performs no network calls."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        dimensions: int | None = None,
        batch_size: int = 64,
        model_identity: str | None = None,
        query_instruction: str | None = None,
    ) -> None:
        if not model or batch_size < 1 or (dimensions is not None and dimensions < 1):
            raise ValueError("model, positive batch_size, and positive dimensions are required")
        self.client, self.model = client, model
        self.dimensions, self.batch_size = dimensions, batch_size
        self.model_identity, self.query_instruction = model_identity, query_instruction

    def _embed(self, inputs: list[Any], task: str) -> list[list[float]]:
        result: list[list[float]] = []
        if inputs and callable(getattr(self.client, "models", None)):
            validate_model_info(
                self.client, self.model, self.dimensions, "text" if isinstance(inputs[0], str) else "image"
            )
        dimension = self.dimensions
        for start in range(0, len(inputs), self.batch_size):
            batch = inputs[start : start + self.batch_size]
            body: dict[str, Any] = {
                "model": self.model,
                "input": batch,
                "task_type": task,
                "encoding_format": "float",
                "error_policy": "fail_fast",
            }
            if self.dimensions is not None:
                body["dimensions"] = self.dimensions
            if self.model_identity:
                body["model_identity"] = self.model_identity
            if task == "RETRIEVAL_QUERY" and self.query_instruction:
                body["instruction"] = self.query_instruction
            response = response_dict(self.client.embed(body))
            if response.get("errors"):
                raise ValueError(f"Antfly returned embedding item failures: {response['errors']}")
            if self.model_identity and response.get("model_identity") != self.model_identity:
                raise ValueError("Antfly embedding model identity did not match the requested pin")
            data = response.get("data", [])
            if len(data) != len(batch) or sorted(item.get("index", -1) for item in data) != list(range(len(batch))):
                raise ValueError("Embedding response must contain each input index exactly once")
            vectors = validate_vectors(
                [item.get("embedding") for item in sorted(data, key=lambda x: x["index"])], len(batch), dimension
            )
            if vectors:
                dimension = len(vectors[0])
            result.extend(vectors)
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("embed_documents accepts text strings")
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("embed_query accepts a text string")
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await run_async(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await run_async(self.embed_query, text)

    def _require_image_model(self) -> None:
        validate_model_info(self.client, self.model, self.dimensions, "image")

    def embed_images(self, images: Sequence[bytes | str | Path], *, mime_type: str = "image/png") -> list[list[float]]:
        """Embed images independently, with advertised image capability validation."""
        if not images:
            return []
        self._require_image_model()
        # Flat content parts have one vector per part in Antfly's public contract.
        return self._embed([image_part(image, mime_type) for image in images], "RETRIEVAL_DOCUMENT")

    def embed_image_query(self, image: bytes | str | Path, *, mime_type: str = "image/png") -> list[float]:
        self._require_image_model()
        return self._embed([image_part(image, mime_type)], "RETRIEVAL_QUERY")[0]

    async def aembed_images(self, images: Sequence[bytes | str | Path], **kwargs: Any) -> list[list[float]]:
        return await run_async(self.embed_images, images, **kwargs)

    async def aembed_image_query(self, image: bytes | str | Path, **kwargs: Any) -> list[float]:
        return await run_async(self.embed_image_query, image, **kwargs)
