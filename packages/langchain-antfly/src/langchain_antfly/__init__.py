"""Antfly integrations for LangChain. Includes graph ingestion values and bounded graph tools."""

from ._utils import WriteOutcomeError
from .embeddings import AntflyEmbeddings
from .graph_documents import GraphDocument, Node, Relationship
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
    "GraphDocument",
    "Node",
    "Relationship",
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
