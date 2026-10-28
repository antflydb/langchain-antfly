"""Antfly integrations for LangChain. Graph dependencies are optional."""

from ._utils import WriteOutcomeError
from .embeddings import AntflyEmbeddings
from .rerankers import AntflyReranker
from .retrievers import AntflyRetriever
from .stores import AntflyByteStore, AntflyDocumentStore, AntflyJSONStore
from .vectorstores import AntflyVectorStore

__all__ = [
    "AntflyEmbeddings",
    "AntflyVectorStore",
    "AntflyRetriever",
    "AntflyReranker",
    "AntflyJSONStore",
    "AntflyDocumentStore",
    "AntflyByteStore",
    "WriteOutcomeError",
    "AntflyGraphStore",
    "create_graph_tools",
]


def __getattr__(name):
    if name == "AntflyGraphStore":
        from .graphs import AntflyGraphStore

        return AntflyGraphStore
    if name == "create_graph_tools":
        from .tools import create_graph_tools

        return create_graph_tools
    raise AttributeError(name)
