"""Graph input compatibility without a provider-specific graph dependency."""

import pytest
from langchain_antfly import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from pydantic import ValidationError


def test_graph_document_roundtrip_preserves_ids_source_and_properties():
    left = Node(id=7, properties={"labels": ["café"]})
    right = Node(id="7", type="Report")
    document = GraphDocument(
        nodes=[left, right],
        relationships=[Relationship(source=left, target=right, type="CITES", properties={"id": "claim-1"})],
        source=Document(id="source", page_content="A café report.", metadata={"id": "stable-source"}),
    )
    restored = GraphDocument.model_validate_json(document.model_dump_json())
    assert restored == document
    assert isinstance(restored.nodes[0].id, int)
    assert isinstance(restored.nodes[1].id, str)
    assert restored.source.metadata["id"] == "stable-source"
    assert restored.relationships[0].properties == {"id": "claim-1"}


def test_graph_document_requires_a_source_for_ownership():
    with pytest.raises(ValidationError):
        GraphDocument(nodes=[], relationships=[])
