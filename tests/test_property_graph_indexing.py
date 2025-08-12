"""Tests for PropertyGraph indexing demonstrating knowledge graph concepts."""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.indices.property_graph.transformations import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)

from fm_app_toolkit.indexing import PropertyGraphIndexer
from fm_app_toolkit.testing.mocks import MockLLMWithChain


def test_property_graph_indexer_creates_index():
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
    assert hasattr(index, 'property_graph_store')


def test_property_graph_with_custom_extractors():
    """Demonstrate configuring custom knowledge extractors.
    
    Different extractors can be used to control how entities and
    relationships are identified in documents.
    """
    documents = [Document(text="Custom extraction of domain-specific entities", doc_id="1")]
    
    # Use only implicit extractor (no LLM required)
    custom_extractors = [ImplicitPathExtractor()]
    
    indexer = PropertyGraphIndexer(
        kg_extractors=custom_extractors,
        show_progress=False,
        embed_kg_nodes=False
    )
    mock_embed = MockEmbedding(embed_dim=256)
    
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)


def test_property_graph_extractor_selection():
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
        embed_kg_nodes=False
    )
    
    with patch.object(PropertyGraphIndex, 'from_documents') as mock_from_docs:
        mock_from_docs.return_value = MagicMock(spec=PropertyGraphIndex)
        indexer_with_llm.create_index(documents, embed_model=mock_embed)
        
        call_args = mock_from_docs.call_args
        kg_extractors = call_args[1]['kg_extractors']
        
        # Should have both extractor types when LLM is available
        assert len(kg_extractors) == 2
        assert any(isinstance(e, SimpleLLMPathExtractor) for e in kg_extractors)
        assert any(isinstance(e, ImplicitPathExtractor) for e in kg_extractors)
    
    # Test WITHOUT LLM - should only use ImplicitPathExtractor
    indexer_without_llm = PropertyGraphIndexer(
        llm=None,
        kg_extractors=None,
        embed_kg_nodes=False
    )
    
    with patch.object(PropertyGraphIndex, 'from_documents') as mock_from_docs:
        mock_from_docs.return_value = MagicMock(spec=PropertyGraphIndex)
        indexer_without_llm.create_index(documents, embed_model=mock_embed)
        
        call_args = mock_from_docs.call_args
        kg_extractors = call_args[1]['kg_extractors']
        
        # Should only have implicit extractor when no LLM
        assert len(kg_extractors) == 1
        assert isinstance(kg_extractors[0], ImplicitPathExtractor)


def test_property_graph_with_embeddings():
    """Demonstrate knowledge graph with embedded nodes for similarity search.
    
    Embedding KG nodes enables vector similarity search over the graph structure,
    combining benefits of both graph and vector representations.
    """
    documents = [
        Document(text="Knowledge graphs can be enhanced with embeddings", doc_id="1"),
    ]
    
    indexer = PropertyGraphIndexer(
        embed_kg_nodes=True,  # Enable embedding of graph nodes
        show_progress=False
    )
    mock_embed = MockEmbedding(embed_dim=256)
    
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)


def test_property_graph_error_handling():
    """Demonstrate error handling in property graph creation.
    
    Errors are logged and propagated for proper application handling.
    """
    documents = [Document(text="Test document", doc_id="test1")]
    indexer = PropertyGraphIndexer()
    
    with patch.object(
        PropertyGraphIndex,
        'from_documents',
        side_effect=Exception("Graph creation failed")
    ):
        with pytest.raises(Exception) as exc_info:
            indexer.create_index(documents)
        assert "Graph creation failed" in str(exc_info.value)