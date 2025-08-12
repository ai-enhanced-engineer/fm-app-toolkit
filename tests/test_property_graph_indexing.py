"""Tests for property graph indexing module."""

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
    """Test PropertyGraphIndexer creates index from documents."""
    documents = [
        Document(text="Alice knows Bob", doc_id="1"),
        Document(text="Bob works at Company X", doc_id="2"),
        Document(text="Company X is in New York", doc_id="3"),
    ]
    
    indexer = PropertyGraphIndexer(show_progress=False, embed_kg_nodes=False)
    mock_embed = MockEmbedding(embed_dim=256)
    
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)


def test_property_graph_indexer_with_llm():
    """Test PropertyGraphIndexer with LLM for extraction."""
    documents = [
        Document(text="Alice is friends with Bob", doc_id="1"),
    ]
    
    # Create mock LLM with predefined responses
    mock_llm = MockLLMWithChain(chain=["Extracted paths"])
    
    indexer = PropertyGraphIndexer(
        llm=mock_llm,
        show_progress=False,
        embed_kg_nodes=False
    )
    mock_embed = MockEmbedding(embed_dim=256)
    
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)


def test_property_graph_indexer_with_custom_extractors():
    """Test PropertyGraphIndexer with custom extractors."""
    documents = [
        Document(text="Test document", doc_id="1"),
    ]
    
    # Create custom extractors
    custom_extractors = [ImplicitPathExtractor()]
    
    indexer = PropertyGraphIndexer(
        kg_extractors=custom_extractors,
        show_progress=False,
        embed_kg_nodes=False
    )
    mock_embed = MockEmbedding(embed_dim=256)
    
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)


def test_property_graph_indexer_with_embedding():
    """Test PropertyGraphIndexer with knowledge graph node embedding."""
    documents = [
        Document(text="Data flows from source to destination", doc_id="1"),
    ]
    
    indexer = PropertyGraphIndexer(
        embed_kg_nodes=True,
        show_progress=False
    )
    mock_embed = MockEmbedding(embed_dim=256)
    
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)


def test_property_graph_indexer_empty_documents():
    """Test PropertyGraphIndexer handles empty document list."""
    documents = []
    
    indexer = PropertyGraphIndexer(embed_kg_nodes=False)
    mock_embed = MockEmbedding(embed_dim=256)
    
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)


def test_property_graph_indexer_with_metadata():
    """Test PropertyGraphIndexer preserves document metadata."""
    documents = [
        Document(
            text="Document with metadata",
            doc_id="meta1",
            metadata={"source": "test.txt", "page": 1}
        ),
        Document(
            text="Another document",
            doc_id="meta2",
            metadata={"source": "test2.txt", "author": "Test Author"}
        ),
    ]
    
    indexer = PropertyGraphIndexer(embed_kg_nodes=False)
    mock_embed = MockEmbedding(embed_dim=256)
    
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)


def test_property_graph_indexer_error_handling():
    """Test PropertyGraphIndexer handles errors appropriately."""
    documents = [
        Document(text="Test document", doc_id="test1"),
    ]
    
    indexer = PropertyGraphIndexer()
    
    # Mock PropertyGraphIndex.from_documents to raise an exception
    with patch.object(
        PropertyGraphIndex,
        'from_documents',
        side_effect=Exception("Index creation failed")
    ):
        with pytest.raises(Exception) as exc_info:
            indexer.create_index(documents)
        assert "Index creation failed" in str(exc_info.value)


def test_property_graph_indexer_with_progress():
    """Test PropertyGraphIndexer with progress display enabled."""
    documents = [
        Document(text=f"Document {i}", doc_id=f"doc_{i}")
        for i in range(3)
    ]
    
    indexer = PropertyGraphIndexer(
        show_progress=True,
        embed_kg_nodes=False
    )
    mock_embed = MockEmbedding(embed_dim=256)
    
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, PropertyGraphIndex)


def test_property_graph_indexer_with_llm_extractors():
    """Test PropertyGraphIndexer creates LLM extractors when LLM provided."""
    documents = [
        Document(text="Complex relationship text", doc_id="1"),
    ]
    
    mock_llm = MockLLMWithChain(chain=["Extracted"])
    
    # Create indexer with LLM but no custom extractors
    indexer = PropertyGraphIndexer(
        llm=mock_llm,
        kg_extractors=None,  # Will use defaults with LLM
        embed_kg_nodes=False
    )
    mock_embed = MockEmbedding(embed_dim=256)
    
    with patch.object(PropertyGraphIndex, 'from_documents') as mock_from_docs:
        mock_from_docs.return_value = MagicMock(spec=PropertyGraphIndex)
        
        indexer.create_index(documents, embed_model=mock_embed)
        
        # Verify that extractors were created with LLM
        call_args = mock_from_docs.call_args
        kg_extractors = call_args[1]['kg_extractors']
        
        # Should have SimpleLLMPathExtractor and ImplicitPathExtractor
        assert len(kg_extractors) == 2
        assert any(isinstance(e, SimpleLLMPathExtractor) for e in kg_extractors)
        assert any(isinstance(e, ImplicitPathExtractor) for e in kg_extractors)


def test_property_graph_indexer_without_llm():
    """Test PropertyGraphIndexer without LLM uses only implicit extractor."""
    documents = [
        Document(text="Simple text", doc_id="1"),
    ]
    
    # Create indexer without LLM and without custom extractors
    indexer = PropertyGraphIndexer(
        llm=None,
        kg_extractors=None,
        embed_kg_nodes=False
    )
    mock_embed = MockEmbedding(embed_dim=256)
    
    with patch.object(PropertyGraphIndex, 'from_documents') as mock_from_docs:
        mock_from_docs.return_value = MagicMock(spec=PropertyGraphIndex)
        
        indexer.create_index(documents, embed_model=mock_embed)
        
        # Verify that only ImplicitPathExtractor was used
        call_args = mock_from_docs.call_args
        kg_extractors = call_args[1]['kg_extractors']
        
        # Should only have ImplicitPathExtractor
        assert len(kg_extractors) == 1
        assert isinstance(kg_extractors[0], ImplicitPathExtractor)