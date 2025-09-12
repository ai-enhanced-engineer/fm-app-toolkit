"""Tests for PropertyGraph indexing demonstrating knowledge graph concepts."""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.indices.property_graph.transformations import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from pydantic import ValidationError

from src.indexing import PropertyGraphIndexer
from src.testing.mock_chain import MockLLMWithChain


def test__property_graph_indexer__creates_index():
    """Demonstrate creating a knowledge graph from documents with relationships.

    PropertyGraphIndex extracts entities and relationships to build
    a structured graph representation of document content.
    """
    # Documents with clear entity relationships
    documents = [
        Document(text="RAG pipelines retrieve context from vector databases", doc_id="1"),
        Document(text="Vector databases store embeddings for similarity search", doc_id="2"),
        Document(text="Embeddings are created by transformer models", doc_id="3"),
    ]

    indexer = PropertyGraphIndexer(show_progress=False, embed_kg_nodes=False)
    mock_embed = MockEmbedding(embed_dim=256)

    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)
    assert hasattr(index, "property_graph_store")


def test__property_graph_with__custom_extractors():
    """Demonstrate configuring custom knowledge extractors.

    Different extractors can be used to control how entities and
    relationships are identified in documents.
    """
    documents = [Document(text="Custom extraction of domain-specific entities", doc_id="1")]

    # Use only implicit extractor (no LLM required)
    custom_extractors = [ImplicitPathExtractor()]

    indexer = PropertyGraphIndexer(kg_extractors=custom_extractors, show_progress=False, embed_kg_nodes=False)
    mock_embed = MockEmbedding(embed_dim=256)

    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)


def test__property_graph_extractor__selection():
    """Demonstrate automatic extractor selection based on LLM availability.

    When an LLM is provided, more sophisticated extraction is possible.
    Without an LLM, only implicit extraction is used.
    """
    documents = [Document(text="Test extractor selection logic", doc_id="1")]
    mock_embed = MockEmbedding(embed_dim=256)

    # Test WITH LLM - should use both SimpleLLMPathExtractor and ImplicitPathExtractor
    mock_llm = MockLLMWithChain(chain=["Extracted entities"])
    indexer_with_llm = PropertyGraphIndexer(
        llm=mock_llm,
        kg_extractors=None,  # Let it auto-select
        embed_kg_nodes=False,
    )

    with patch.object(PropertyGraphIndex, "from_documents") as mock_from_docs:
        mock_from_docs.return_value = MagicMock(spec=PropertyGraphIndex)
        indexer_with_llm.create_index(documents, embed_model=mock_embed)

        call_args = mock_from_docs.call_args
        kg_extractors = call_args[1]["kg_extractors"]

        # Should have both extractor types when LLM is available
        assert len(kg_extractors) == 2
        assert any(isinstance(e, SimpleLLMPathExtractor) for e in kg_extractors)
        assert any(isinstance(e, ImplicitPathExtractor) for e in kg_extractors)

    # Test WITHOUT LLM - should only use ImplicitPathExtractor
    indexer_without_llm = PropertyGraphIndexer(llm=None, kg_extractors=None, embed_kg_nodes=False)

    with patch.object(PropertyGraphIndex, "from_documents") as mock_from_docs:
        mock_from_docs.return_value = MagicMock(spec=PropertyGraphIndex)
        indexer_without_llm.create_index(documents, embed_model=mock_embed)

        call_args = mock_from_docs.call_args
        kg_extractors = call_args[1]["kg_extractors"]

        # Should only have implicit extractor when no LLM
        assert len(kg_extractors) == 1
        assert isinstance(kg_extractors[0], ImplicitPathExtractor)


def test__property_graph_with__embeddings():
    """Demonstrate knowledge graph with embedded nodes for similarity search.

    Embedding KG nodes enables vector similarity search over the graph structure,
    combining benefits of both graph and vector representations.
    """
    documents = [
        Document(text="Knowledge graphs can be enhanced with embeddings", doc_id="1"),
    ]

    indexer = PropertyGraphIndexer(
        embed_kg_nodes=True,  # Enable embedding of graph nodes
        show_progress=False,
    )
    mock_embed = MockEmbedding(embed_dim=256)

    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)


def test__property_graph_error__handling():
    """Demonstrate error handling in property graph creation.

    Errors are logged and propagated for proper application handling.
    """
    documents = [Document(text="Test document", doc_id="test1")]
    indexer = PropertyGraphIndexer()

    with patch.object(PropertyGraphIndex, "from_documents", side_effect=Exception("Graph creation failed")):
        with pytest.raises(Exception) as exc_info:
            indexer.create_index(documents)
        assert "Graph creation failed" in str(exc_info.value)


def test__property_graph_indexer__validates_input_types():
    """Demonstrate that Pydantic validates input types for PropertyGraphIndexer.

    The @validate_call decorator provides consistent validation across all
    indexer implementations, ensuring type safety and clear error messages.
    """
    indexer = PropertyGraphIndexer(embed_kg_nodes=False)
    mock_embed = MockEmbedding(embed_dim=256)

    # Invalid: string instead of list
    with pytest.raises(ValidationError) as exc_info:
        indexer.create_index("not a list", embed_model=mock_embed)
    assert "Input should be a valid list" in str(exc_info.value)

    # Invalid: None instead of list
    with pytest.raises(ValidationError) as exc_info:
        indexer.create_index(None, embed_model=mock_embed)
    assert "Input should be a valid list" in str(exc_info.value)

    # Invalid: dict instead of list
    with pytest.raises(ValidationError) as exc_info:
        indexer.create_index({"doc": "value"}, embed_model=mock_embed)
    assert "Input should be a valid list" in str(exc_info.value)

    # Invalid: integer instead of list
    with pytest.raises(ValidationError) as exc_info:
        indexer.create_index(42, embed_model=mock_embed)
    assert "Input should be a valid list" in str(exc_info.value)

    # Valid: empty list should work
    index = indexer.create_index([], embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)

    # Valid: list with Documents should work
    docs = [Document(text="Knowledge graph test", doc_id="kg1")]
    index = indexer.create_index(docs, embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)

    # Note: Pydantic's @validate_call automatically converts tuples to lists,
    # which is acceptable behavior for our use case
