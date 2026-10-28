"""Native dense, text, and hybrid retrieval with optional reranking."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, model_validator

from ._utils import hits, run_async
from .embeddings import validate_vectors
from .rerankers import AntflyReranker
from .vectorstores import AntflyVectorStore, combine_filters


class AntflyRetriever(BaseRetriever):
    """Expose Antfly retrieval without changing VectorStore distance semantics.

    ``filter`` is a fixed scope. Per-call ``filter`` is intersected with it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    vectorstore: AntflyVectorStore | None = Field(default=None, exclude=True)
    client: Any = Field(default=None, exclude=True)
    table: str | None = None
    index: str = "vectors"
    content_field: str = "text"
    metadata_field: str = "metadata"
    mode: Literal["dense", "text", "hybrid"] = "hybrid"
    k: int = 4
    fetch_k: int = 20
    filter: dict[str, Any] | None = None
    full_text_index: str | None = None
    merge_config: dict[str, Any] = Field(default_factory=lambda: {"strategy": "rrf"})
    reranker: AntflyReranker | None = None
    graph_store: Any = Field(default=None, exclude=True)
    graph_expansion: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _options(self) -> AntflyRetriever:
        if self.vectorstore is not None and (self.client is not None or self.table is not None):
            raise ValueError("Pass vectorstore or client/table, not both")
        if self.vectorstore is None and (self.client is None or not self.table):
            raise ValueError("Provide vectorstore or client and table")
        if self.k < 0 or self.fetch_k < self.k:
            raise ValueError("Require 0 <= k <= fetch_k")
        if (self.graph_store is None) != (self.graph_expansion is None):
            raise ValueError("graph_store and graph_expansion must be configured together")
        if self.graph_expansion:
            unknown = set(self.graph_expansion) - {"seed_keys", "max_depth", "limit", "edge_types"}
            if unknown:
                raise ValueError(f"Unsupported graph expansion options: {', '.join(sorted(unknown))}")
        return self

    def _store(self) -> AntflyVectorStore:
        return self.vectorstore or AntflyVectorStore(
            client=self.client,
            table=self.table,
            index=self.index,
            content_field=self.content_field,
            metadata_field=self.metadata_field,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, filter: dict[str, Any] | None = None
    ) -> list[Document]:
        if self.k == 0:
            return []
        store = self._store()
        scope = combine_filters(store.filter, combine_filters(self.filter, filter))
        count = self.fetch_k if self.reranker or self.graph_expansion else self.k
        if count == 0:
            return []
        options: dict[str, Any] = {
            "limit": count,
            "filter_query": scope,
            "fields": [store.content_field, store.metadata_field],
        }
        if self.mode in {"dense", "hybrid"}:
            if store.embeddings is not None:
                store._ensure_index(refresh=True, allow_deferred=True)
                vector = validate_vectors([store.embeddings.embed_query(query)], 1, store.dimension)[0]
                store._ensure_index(len(vector))
                options["embeddings"] = {store.index: vector}
            else:
                store._ensure_index(refresh=True)
                options["semantic_search"] = query
            options["indexes"] = [store.index]
        if self.mode in {"text", "hybrid"}:
            options["full_text_search"] = {"match": query, "field": store.content_field}
            if self.full_text_index:
                options["full_text_index"] = self.full_text_index
        if self.mode == "hybrid":
            options["merge_config"] = self.merge_config
        documents = [store._document(hit, provenance=True) for hit in hits(store.client.query(store.table, **options))]
        if self.graph_expansion:
            documents = self.graph_store.expand_documents(documents, filter_query=scope, **self.graph_expansion)
        if self.reranker:
            documents = self.reranker.rerank(query, documents)
        return documents[: self.k]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, filter: dict[str, Any] | None = None
    ) -> list[Document]:
        return await run_async(self._get_relevant_documents, query, run_manager=run_manager, filter=filter)
