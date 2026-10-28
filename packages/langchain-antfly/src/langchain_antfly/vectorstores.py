"""Antfly's dense VectorStore, with explicit managed/external embedding modes."""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ._utils import hits, response_dict, run_async, write
from .embeddings import validate_model_info, validate_vectors


def _status(error: Exception, code: int) -> bool:
    return getattr(error, "status_code", None) == code


def combine_filters(fixed: dict[str, Any] | None, supplied: dict[str, Any] | None) -> dict[str, Any] | None:
    if fixed and supplied:
        return {"conjuncts": [fixed, supplied]}
    return fixed or supplied


class AntflyVectorStore(VectorStore):
    """Attach without I/O. Use create/from_documents to provision resources explicitly.

    ``embedding=None`` means the existing index generates embeddings on the server.
    An ordinary LangChain Embeddings instance selects external-vector mode.
    ``filter`` is a read scope for search and get_by_ids, not mutation authorization.
    Explicit-ID writes/deletes use the server credentials and authorization rules.
    """

    def __init__(
        self,
        *,
        client: Any,
        table: str,
        index: str = "vectors",
        embedding: Embeddings | None = None,
        content_field: str = "text",
        metadata_field: str = "metadata",
        distance_metric: str | None = None,
        dimension: int | None = None,
        filter: dict[str, Any] | None = None,
        wait: bool = True,
        embedder: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        if not table or not index or not content_field or not metadata_field:
            raise ValueError("table, index, content_field and metadata_field are required")
        if content_field == metadata_field or content_field.startswith("_") or metadata_field.startswith("_"):
            raise ValueError("Content and metadata fields must be distinct ordinary document fields")
        if distance_metric is not None and distance_metric not in {"cosine", "l2_squared", "inner_product"}:
            raise ValueError("distance_metric must be cosine, l2_squared, or inner_product")
        if embedding is not None and embedder is not None:
            raise ValueError("embedding (external) and embedder (managed) are mutually exclusive")
        if dimension is not None and dimension < 1:
            raise ValueError("dimension must be positive")
        self.client, self.table, self.index = client, table, index
        self._embedding, self.embedder = embedding, embedder
        self.content_field, self.metadata_field = content_field, metadata_field
        self._metric_explicit = distance_metric is not None
        self.distance_metric, self.dimension = distance_metric or "cosine", dimension
        self.filter, self.wait, self.provenance = filter, wait, provenance
        self._provision = False
        self._validated = False

    @property
    def embeddings(self) -> Embeddings | None:
        return self._embedding

    def _config(self, dimension: int | None = None) -> dict[str, Any]:
        config: dict[str, Any] = {
            "type": "embeddings",
            "distance_metric": self.distance_metric,
            "description": "langchain-antfly:"
            + json.dumps(
                {
                    "version": 1,
                    "content_field": self.content_field,
                    "metadata_field": self.metadata_field,
                    "provenance": self.provenance,
                },
                sort_keys=True,
                allow_nan=False,
            ),
        }
        if self._embedding is not None:
            config["external"] = True
            if dimension is None:
                dimension = self.dimension
            if dimension is None:
                raise ValueError("External index creation requires dimension or actual ingestion vectors")
            config["dimension"] = dimension
        else:
            if not self.embedder:
                raise ValueError("Creating a managed index requires an explicit embedder configuration")
            config.update(field=self.content_field, embedder=self.embedder)
            if self.dimension is not None:
                config["dimension"] = self.dimension
        return config

    @classmethod
    def create(cls, *, client: Any, table: str, **kwargs: Any) -> AntflyVectorStore:
        store = cls(client=client, table=table, **kwargs)
        # An external provider needs no hidden probe: defer index creation until ingestion learns shape.
        config = store._config() if store._embedding is None or store.dimension is not None else None
        client.create_table(table, indexes={store.index: config} if config is not None else None)
        store._provision = True
        return store

    def _ensure_index(
        self, dimension: int | None = None, *, refresh: bool = False, allow_deferred: bool = False
    ) -> None:
        if refresh:
            self._validated = False
        if self._validated:
            if dimension is not None and self.dimension is not None and dimension != self.dimension:
                raise ValueError(f"Embedding dimension mismatch: index expects {self.dimension}, received {dimension}")
            return
        try:
            info = response_dict(self.client.indexes.get(self.table, self.index))
        except Exception as error:
            if not _status(error, 404) or not self._provision:
                raise
            # A missing index and a missing table must remain distinguishable.
            self.client.get_table(self.table)
            if allow_deferred and self._embedding is not None and dimension is None and self.dimension is None:
                return
            config = self._config(dimension)
            try:
                self.client.indexes.create(self.table, self.index, config)
            except Exception as race:
                if not _status(race, 409):
                    raise
            info = response_dict(self.client.indexes.get(self.table, self.index))
        config = info.get("config", info)
        if config.get("type", "embeddings") != "embeddings" or config.get("sparse", False):
            raise ValueError("AntflyVectorStore requires a dense embeddings index")
        if bool(config.get("external", False)) != (self._embedding is not None):
            raise ValueError("Index embedding mode conflicts with the supplied external embedding provider")
        metric = config.get("distance_metric", "l2_squared")
        if self._metric_explicit and metric != self.distance_metric:
            raise ValueError(f"Index distance metric is {metric!r}; configure distance_metric={metric!r}")
        self.distance_metric = metric
        description = config.get("description", "")
        if description.startswith("langchain-antfly:"):
            declaration = json.loads(description.removeprefix("langchain-antfly:"))
            for field in ("content_field", "metadata_field"):
                if declaration.get(field) != getattr(self, field):
                    raise ValueError(f"Existing adapter index has incompatible {field}")
            known = declaration.get("provenance") or {}
            for key, value in (self.provenance or {}).items():
                if key in known and known[key] != value:
                    raise ValueError(f"External embedding provenance conflict for {key!r}")
        expected = config.get("dimension")
        wanted = dimension or self.dimension
        if wanted is not None and expected is not None and wanted != expected:
            raise ValueError(f"Embedding dimension mismatch: index expects {expected}, received {wanted}")
        if self._embedding is None:
            field = config.get("field")
            if field is not None and field != self.content_field:
                raise ValueError("Managed index content field differs from content_field")
            if self.embedder:
                actual = config.get("embedder", {})
                for key, value in self.embedder.items():
                    # Public index configuration deliberately omits write-only credentials.
                    if key in {"api_key", "credentials_path"}:
                        continue
                    if actual.get(key) != value:
                        raise ValueError(f"Managed embedder {key!r} conflicts with the existing index")
        self.dimension = expected or wanted
        actual_embedder = config.get("embedder", {})
        if self._embedding is None and actual_embedder.get("provider") == "antfly":
            # A remote inference endpoint has its own registry, unavailable via this client.
            if not actual_embedder.get("api_url") and callable(getattr(self.client, "models", None)):
                validate_model_info(self.client, actual_embedder.get("model"), self.dimension, "text")
        self._validated = True

    def validate(self) -> None:
        """Validate existing index compatibility without changing resources."""
        self._validated = False
        self._ensure_index()

    def add_texts(
        self, texts: Iterable[str], metadatas: list[dict] | None = None, *, ids: list[str] | None = None, **kwargs: Any
    ) -> list[str]:
        vectors = kwargs.pop("_vectors", None)
        wait = kwargs.pop("wait", self.wait)
        if kwargs:
            raise TypeError(f"Unsupported add options: {', '.join(kwargs)}")
        texts = list(texts)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must contain strings")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas length must equal texts length")
        if ids is not None and len(ids) != len(texts):
            raise ValueError("ids length must equal texts length")
        keys = [key if key is not None else str(uuid.uuid4()) for key in (ids or [None] * len(texts))]
        if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("IDs must be unique nonempty strings within a batch")
        if not texts:
            return []
        metadata_values = []
        for metadata in metadatas if metadatas is not None else [{} for _ in texts]:
            if not isinstance(metadata, dict):
                raise TypeError("metadatas must contain dictionaries")
            if "_antfly" in metadata:
                raise ValueError("metadata._antfly is reserved for adapter provenance")
            metadata_values.append(json.loads(json.dumps(metadata, allow_nan=False)))
        self._ensure_index(refresh=True, allow_deferred=True)
        if self._embedding is not None:
            vectors = validate_vectors(
                vectors if vectors is not None else self._embedding.embed_documents(texts), len(texts), self.dimension
            )
            self._ensure_index(len(vectors[0]))
        else:
            if vectors is not None:
                raise ValueError("Managed indexes do not accept external vectors")
            self._ensure_index()
        inserts = {}
        for position, (key, text) in enumerate(zip(keys, texts, strict=True)):
            metadata = metadata_values[position]
            document: dict[str, Any] = {self.content_field: text, self.metadata_field: metadata}
            if vectors is not None:
                document["_embeddings"] = {self.index: vectors[position]}
            inserts[key] = document
        write(self.client, self.table, inserts=inserts, wait=wait)
        return keys

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        ids = kwargs.pop("ids", None)
        if ids is None:
            ids = [document.id for document in documents]
        return self.add_texts([d.page_content for d in documents], [d.metadata for d in documents], ids=ids, **kwargs)

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        """Delete explicit IDs using server authorization; read filters do not restrict writes."""
        if kwargs:
            raise TypeError(f"Unsupported delete options: {', '.join(kwargs)}")
        if ids is None:
            raise ValueError("Explicit document IDs are required; deletion never drops a table")
        if ids:
            write(self.client, self.table, deletes=ids, wait=self.wait)
        return True

    def _document(self, hit: dict[str, Any], *, provenance: bool = False) -> Document:
        source = hit.get("_source", hit.get("source", {}))
        metadata = dict(source.get(self.metadata_field, {}))
        if "_antfly" in metadata:
            raise ValueError("Stored metadata._antfly conflicts with adapter provenance")
        if provenance:
            metadata["_antfly"] = {
                key.removeprefix("_"): hit[key]
                for key in ("_score", "_distance", "_table", "_index_scores")
                if key in hit
            }
        return Document(
            id=hit.get("_id", hit.get("id")), page_content=source.get(self.content_field, ""), metadata=metadata
        )

    def get_by_ids(self, ids: Sequence[str]) -> list[Document]:
        if ids:
            self.client.get_table(self.table)
        result = []
        for key in dict.fromkeys(ids):
            if self.filter:
                found = hits(
                    self.client.query(
                        self.table,
                        limit=1,
                        fields=[self.content_field, self.metadata_field],
                        filter_query=combine_filters(self.filter, {"ids": [key]}),
                    )
                )
                result.extend(self._document(hit) for hit in found)
                continue
            try:
                source = response_dict(self.client.get(self.table, key))
            except Exception as error:
                if _status(error, 404):
                    # Do not turn a concurrently dropped table into missing keys.
                    self.client.get_table(self.table)
                    continue
                raise
            result.append(self._document({"_id": key, "_source": source}))
        return result

    def _search_hits(
        self,
        query: str | None,
        vector: list[float] | None,
        k: int,
        *,
        filter: dict[str, Any] | None = None,
        include_vectors: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if k < 0:
            raise ValueError("k must be nonnegative")
        if not k:
            return []
        allowed = {"search_effort", "timeout_ms", "distance_under", "distance_over"}
        if set(kwargs) - allowed:
            raise TypeError(f"Unsupported search options: {', '.join(sorted(set(kwargs) - allowed))}")
        # Refresh compatibility before any provider invocation. A dropped or
        # reconfigured resource must not trigger an unnecessary inference call.
        self._ensure_index(
            len(vector) if vector is not None else None,
            refresh=True,
            allow_deferred=vector is None and self._embedding is not None,
        )
        if vector is None and self._embedding is not None:
            vector = validate_vectors([self._embedding.embed_query(query)], 1, self.dimension)[0]
        if vector is not None:
            vector = validate_vectors([vector], 1, self.dimension)[0]
        self._ensure_index(len(vector) if vector is not None else None)
        fields = [self.content_field, self.metadata_field]
        if include_vectors:
            fields.append("_embeddings")
        response = self.client.query(
            self.table,
            semantic_search=query if vector is None else None,
            embeddings={self.index: vector} if vector is not None else None,
            indexes=[self.index],
            fields=fields,
            limit=k,
            filter_query=combine_filters(self.filter, filter),
            **kwargs,
        )
        return hits(response)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return [self._document(hit) for hit in self._search_hits(query, None, k, **kwargs)]

    def similarity_search_by_vector(self, embedding: list[float], k: int = 4, **kwargs: Any) -> list[Document]:
        return [self._document(hit) for hit in self._search_hits(None, embedding, k, **kwargs)]

    def _with_scores(self, found: list[dict[str, Any]]) -> list[tuple[Document, float]]:
        if any("_distance" not in hit for hit in found):
            raise ValueError("Dense distance is unavailable; fused scores cannot be returned as vector distances")
        return [(self._document(hit), float(hit["_distance"])) for hit in found]

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> list[tuple[Document, float]]:
        return self._with_scores(self._search_hits(query, None, k, **kwargs))

    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        return self._with_scores(self._search_hits(None, embedding, k, **kwargs))

    def _similarity_search_with_relevance_scores(self, query: str, k: int = 4, **kwargs: Any):
        # Attachment learns the metric during search, before selecting its conversion.
        found = self.similarity_search_with_score(query, k, **kwargs)
        convert = self._select_relevance_score_fn()
        return [(document, convert(distance)) for document, distance in found]

    async def _asimilarity_search_with_relevance_scores(self, query: str, k: int = 4, **kwargs: Any):
        return await run_async(self._similarity_search_with_relevance_scores, query, k, **kwargs)

    def _select_relevance_score_fn(self):
        if self.distance_metric == "cosine":
            return lambda distance: max(0.0, min(1.0, 1.0 - distance / 2.0))
        if self.distance_metric == "l2_squared":
            return lambda distance: 1.0 / (1.0 + max(0.0, distance))
        # Antfly inner-product distance is -dot(query, document); logistic maps it monotonically.
        return lambda distance: (
            (math.exp(-distance) / (1.0 + math.exp(-distance))) if distance >= 0 else 1.0 / (1.0 + math.exp(distance))
        )

    def _mmr(
        self, vector: list[float] | None, found: list[dict[str, Any]], k: int, lambda_mult: float
    ) -> list[Document]:
        if not 0 <= lambda_mult <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")
        if not found or k == 0:
            return []
        candidates = []
        for hit in found:
            source = hit.get("_source", {})
            embedding = source.get("_embeddings", {}).get(self.index)
            if embedding is None:
                source = response_dict(self.client.get(self.table, hit["_id"], fields=["_embeddings"]))
                embedding = source.get("_embeddings", {}).get(self.index)
            if isinstance(embedding, dict):
                embedding = embedding.get("embedding", embedding.get("vector"))
            if embedding is None:
                raise ValueError("MMR requires persisted dense vectors; this index did not expose its vectors")
            candidates.append(validate_vectors([embedding], 1, len(vector) if vector else self.dimension)[0])
        matrix = np.asarray(candidates, dtype=float)
        if any("_distance" not in hit for hit in found):
            raise ValueError("MMR requires exact dense distances; fused rankings are unsupported")
        distances = np.asarray([hit["_distance"] for hit in found], dtype=float)
        if self.distance_metric == "cosine":
            normalized = matrix / np.linalg.norm(matrix, axis=1)[:, None]
            pairwise = normalized @ normalized.T
            relevance = 1.0 - distances
        elif self.distance_metric == "l2_squared":
            pairwise = 1.0 / (1.0 + np.sum((matrix[:, None] - matrix[None, :]) ** 2, axis=2))
            relevance = 1.0 / (1.0 + np.maximum(0.0, distances))
        else:
            pairwise = matrix @ matrix.T
            relevance = -distances
        selected = [int(np.argmax(relevance))]
        while len(selected) < min(k, len(found)):
            diversity = np.max(pairwise[:, selected], axis=1)
            scores = lambda_mult * relevance - (1 - lambda_mult) * diversity
            scores[selected] = -np.inf
            selected.append(int(np.argmax(scores)))
        return [self._document(found[position]) for position in selected]

    def max_marginal_relevance_search_by_vector(
        self, embedding: list[float], k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any
    ) -> list[Document]:
        vector = validate_vectors([embedding], 1, self.dimension)[0]
        if k < 0 or fetch_k < k:
            raise ValueError("Require 0 <= k <= fetch_k")
        return self._mmr(
            vector, self._search_hits(None, vector, fetch_k, include_vectors=True, **kwargs), k, lambda_mult
        )

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any
    ) -> list[Document]:
        """Select with index-metric relevance and diversity using persisted vectors.

        Cosine uses cosine similarity, squared L2 uses 1/(1+distance), and inner
        product uses dot product. Managed mode reuses exact query distances;
        no extra embedding call or candidate re-embedding is performed.
        """
        if k < 0 or fetch_k < k:
            raise ValueError("Require 0 <= k <= fetch_k")
        found = self._search_hits(query, None, fetch_k, include_vectors=True, **kwargs)
        return self._mmr(None, found, k, lambda_mult)

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        create_if_missing: bool = True,
        **kwargs: Any,
    ) -> AntflyVectorStore:
        store = cls(embedding=embedding, **kwargs)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must contain strings")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas length must equal texts length")
        if ids is not None and len(ids) != len(texts):
            raise ValueError("ids length must equal texts length")
        if ids is not None:
            present = [key for key in ids if key is not None]
            if any(not isinstance(key, str) or not key for key in present) or len(set(present)) != len(present):
                raise ValueError("IDs must be unique nonempty strings within a batch")
        # Validate all caller data before provisioning or invoking paid inference.
        for metadata in metadatas if metadatas is not None else []:
            if not isinstance(metadata, dict):
                raise TypeError("metadatas must contain dictionaries")
            if "_antfly" in metadata:
                raise ValueError("metadata._antfly is reserved for adapter provenance")
            json.dumps(metadata, allow_nan=False)
        try:
            store.client.get_table(store.table)
        except Exception as error:
            if not _status(error, 404) or not create_if_missing:
                raise
            config = store._config() if embedding is None or store.dimension is not None else None
            try:
                store.client.create_table(store.table, indexes={store.index: config} if config else None)
            except Exception as race:
                if not _status(race, 409):
                    raise
        store._provision = create_if_missing
        if texts or embedding is None or store.dimension is not None:
            store._ensure_index(store.dimension, allow_deferred=True)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls, documents: list[Document], embedding: Embeddings | None = None, **kwargs: Any
    ) -> AntflyVectorStore:
        ids = kwargs.pop("ids", [document.id for document in documents])
        return cls.from_texts(
            [d.page_content for d in documents], embedding, [d.metadata for d in documents], ids=ids, **kwargs
        )

    @classmethod
    async def afrom_texts(
        cls, texts: list[str], embedding: Embeddings | None = None, metadatas: list[dict] | None = None, **kwargs: Any
    ) -> AntflyVectorStore:
        return await run_async(cls.from_texts, texts, embedding, metadatas, **kwargs)

    @classmethod
    async def afrom_documents(
        cls, documents: list[Document], embedding: Embeddings | None = None, **kwargs: Any
    ) -> AntflyVectorStore:
        return await run_async(cls.from_documents, documents, embedding, **kwargs)

    async def aadd_texts(self, texts: Iterable[str], metadatas: list[dict] | None = None, **kwargs: Any) -> list[str]:
        return await run_async(self.add_texts, texts, metadatas, **kwargs)

    async def aadd_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        return await run_async(self.add_documents, documents, **kwargs)

    async def adelete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        return await run_async(self.delete, ids, **kwargs)

    async def aget_by_ids(self, ids: Sequence[str]) -> list[Document]:
        return await run_async(self.get_by_ids, ids)

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return await run_async(self.similarity_search, query, k, **kwargs)

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        return await run_async(self.similarity_search_with_score, query, k, **kwargs)

    async def asimilarity_search_by_vector(self, embedding: list[float], k: int = 4, **kwargs: Any) -> list[Document]:
        return await run_async(self.similarity_search_by_vector, embedding, k, **kwargs)

    async def amax_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any
    ) -> list[Document]:
        return await run_async(self.max_marginal_relevance_search, query, k, fetch_k, lambda_mult, **kwargs)

    async def amax_marginal_relevance_search_by_vector(
        self, embedding: list[float], k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any
    ) -> list[Document]:
        return await run_async(
            self.max_marginal_relevance_search_by_vector, embedding, k, fetch_k, lambda_mult, **kwargs
        )

    def add_images(
        self,
        images: Sequence[bytes | str | Path],
        *,
        ids: list[str] | None = None,
        metadatas: list[dict] | None = None,
        mime_type: str = "image/png",
    ) -> list[str]:
        if self._embedding is None or not callable(getattr(self._embedding, "embed_images", None)):
            raise ValueError("Image ingestion requires an embedding provider with explicit embed_images support")
        if ids is not None:
            if len(ids) != len(images):
                raise ValueError("ids length must equal images length")
            present = [key for key in ids if key is not None]
            if any(not isinstance(key, str) or not key for key in present) or len(set(present)) != len(present):
                raise ValueError("IDs must be unique nonempty strings within a batch")
        if metadatas is not None:
            if len(metadatas) != len(images):
                raise ValueError("metadatas length must equal images length")
            for metadata in metadatas:
                if not isinstance(metadata, dict):
                    raise TypeError("metadatas must contain dictionaries")
                if "_antfly" in metadata:
                    raise ValueError("metadata._antfly is reserved for adapter provenance")
                json.dumps(metadata, allow_nan=False)
        if not images:
            return []
        self._ensure_index(refresh=True, allow_deferred=True)
        vectors = self._embedding.embed_images(images, mime_type=mime_type)
        return self.add_texts(["" for _ in images], metadatas, ids=ids, _vectors=vectors)

    def similarity_search_by_image(
        self, image: bytes | str | Path, k: int = 4, *, mime_type: str = "image/png", **kwargs: Any
    ) -> list[Document]:
        if self._embedding is None or not callable(getattr(self._embedding, "embed_image_query", None)):
            raise ValueError("Image search requires a provider with explicit embed_image_query support")
        self._ensure_index(refresh=True, allow_deferred=True)
        vector = self._embedding.embed_image_query(image, mime_type=mime_type)
        return self.similarity_search_by_vector(vector, k, **kwargs)

    async def aadd_images(self, images: Sequence[bytes | str | Path], **kwargs: Any) -> list[str]:
        return await run_async(self.add_images, images, **kwargs)

    async def asimilarity_search_by_image(self, image: bytes | str | Path, k: int = 4, **kwargs: Any) -> list[Document]:
        return await run_async(self.similarity_search_by_image, image, k, **kwargs)
