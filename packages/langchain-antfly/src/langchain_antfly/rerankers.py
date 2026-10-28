"""Standalone Antfly reranking and LangChain document compression."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.runnables import RunnableLambda
from pydantic import ConfigDict, Field, model_validator

from ._utils import response_dict, run_async


class AntflyReranker(BaseDocumentCompressor):
    """Sort scores returned in input order, preserving user documents and metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    client: Any = Field(exclude=True)
    model: str
    top_n: int | None = None

    @model_validator(mode="after")
    def _validate_options(self) -> AntflyReranker:
        if not self.model or (self.top_n is not None and self.top_n < 0):
            raise ValueError("model is required and top_n must be nonnegative")
        return self

    def rerank(self, query: str, documents: Sequence[Document], *, top_n: int | None = None) -> list[Document]:
        limit = self.top_n if top_n is None else top_n
        if limit is not None and limit < 0:
            raise ValueError("top_n must be nonnegative")
        if not documents or limit == 0:
            return []
        response = response_dict(
            self.client.rerank(
                {"model": self.model, "query": query, "documents": [doc.page_content for doc in documents]}
            )
        )
        data = response.get("data", [])
        if len(data) != len(documents) or sorted(item.get("index", -1) for item in data) != list(range(len(documents))):
            raise ValueError("Reranking response must contain each document index exactly once")
        for item in data:
            score = item.get("score")
            if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score):
                raise ValueError("Reranking scores must be finite numbers")
        data.sort(key=lambda item: (-item["score"], item["index"]))
        result = []
        for item in data[:limit]:
            original = documents[item["index"]]
            metadata = dict(original.metadata)
            reserved = metadata.get("_antfly", {})
            if not isinstance(reserved, dict):
                raise TypeError("metadata._antfly is reserved for Antfly provenance")
            metadata["_antfly"] = {**reserved, "rerank_score": float(item["score"]), "rerank_index": item["index"]}
            result.append(Document(id=original.id, page_content=original.page_content, metadata=metadata))
        return result

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Any = None
    ) -> Sequence[Document]:
        return self.rerank(query, documents)

    async def acompress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Any = None
    ) -> Sequence[Document]:
        return await run_async(self.compress_documents, documents, query, callbacks)

    def invoke(self, input: dict[str, Any], config: Any = None, **kwargs: Any) -> list[Document]:
        if set(input) != {"query", "documents"}:
            raise ValueError("Reranker input requires exactly query and documents")
        return self.rerank(input["query"], input["documents"], **kwargs)

    async def ainvoke(self, input: dict[str, Any], config: Any = None, **kwargs: Any) -> list[Document]:
        return await run_async(self.invoke, input, config, **kwargs)

    def as_runnable(self) -> RunnableLambda:
        return RunnableLambda(self.invoke, afunc=self.ainvoke)

    async def arerank(self, query: str, documents: Sequence[Document], *, top_n: int | None = None) -> list[Document]:
        return await run_async(self.rerank, query, documents, top_n=top_n)
