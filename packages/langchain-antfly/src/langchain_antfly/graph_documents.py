"""Graph ingestion values owned by the Antfly integration.

The field names also match graph documents produced by other integrations.
"""

from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class Node(BaseModel):
    """A graph entity identified by its type and string or integer ID."""

    id: str | int
    type: str = "Node"
    properties: dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    """A directed, typed relationship with application-defined properties."""

    source: Node
    target: Node
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphDocument(BaseModel):
    """Nodes and relationships extracted from a required source Document.

    Set source.metadata["id"] to a stable value for updates and deletion.
    """

    nodes: list[Node]
    relationships: list[Relationship]
    source: Document
