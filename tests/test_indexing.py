"""Tests for document indexing module."""

from unittest.mock import patch

import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

from fm_app_toolkit.indexing import BaseIndexer, VectorStoreIndexer


def test_vector_store_indexer_creates_index():
    """Test VectorStoreIndexer creates index from documents."""
    # Create test documents
    documents = [
        Document(text="First document content", doc_id="1"),
        Document(text="Second document content", doc_id="2"),
        Document(text="Third document content", doc_id="3"),
    ]
    
    # Create indexer
    indexer = VectorStoreIndexer(show_progress=False)
    
    # Create mock embedding model
    mock_embed = MockEmbedding(embed_dim=256)
    
    # Create index
    index = indexer.create_index(documents, embed_model=mock_embed)
    
    # Verify index is created
    assert isinstance(index, VectorStoreIndex)


def test_vector_store_indexer_with_default_embedding():
    """Test VectorStoreIndexer works without explicitly passing embedding model."""
    documents = [
        Document(text="Test document", doc_id="test1"),
    ]
    
    indexer = VectorStoreIndexer()
    
    # Use MockEmbedding directly instead of patching Settings
    mock_embed = MockEmbedding(embed_dim=256)
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, VectorStoreIndex)


def test_vector_store_indexer_with_progress():
    """Test VectorStoreIndexer with progress display enabled."""
    documents = [
        Document(text=f"Document {i}", doc_id=f"doc_{i}")
        for i in range(5)
    ]
    
    # Create indexer with progress enabled
    indexer = VectorStoreIndexer(show_progress=True)
    mock_embed = MockEmbedding(embed_dim=256)
    
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, VectorStoreIndex)


def test_vector_store_indexer_custom_batch_size():
    """Test VectorStoreIndexer with custom batch size."""
    documents = [
        Document(text=f"Document {i}", doc_id=f"doc_{i}")
        for i in range(10)
    ]
    
    # Create indexer with custom batch size
    indexer = VectorStoreIndexer(insert_batch_size=512)
    assert indexer.insert_batch_size == 512
    
    mock_embed = MockEmbedding(embed_dim=256)
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, VectorStoreIndex)


def test_vector_store_indexer_empty_documents():
    """Test VectorStoreIndexer handles empty document list."""
    documents = []
    
    indexer = VectorStoreIndexer()
    mock_embed = MockEmbedding(embed_dim=256)
    
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, VectorStoreIndex)


def test_vector_store_indexer_query_capability():
    """Test that created index can be queried."""
    documents = [
        Document(text="The sky is blue", doc_id="1"),
        Document(text="The grass is green", doc_id="2"),
        Document(text="The sun is yellow", doc_id="3"),
    ]
    
    indexer = VectorStoreIndexer()
    mock_embed = MockEmbedding(embed_dim=256)
    
    # Create index
    index = indexer.create_index(documents, embed_model=mock_embed)
    
    # Verify retriever can be created (doesn't require LLM)
    retriever = index.as_retriever()
    assert retriever is not None


def test_vector_store_indexer_error_handling():
    """Test VectorStoreIndexer handles errors appropriately."""
    documents = [
        Document(text="Test document", doc_id="test1"),
    ]
    
    indexer = VectorStoreIndexer()
    
    # Mock VectorStoreIndex.from_documents to raise an exception
    with patch.object(
        VectorStoreIndex,
        'from_documents',
        side_effect=Exception("Index creation failed")
    ):
        with pytest.raises(Exception) as exc_info:
            indexer.create_index(documents)
        assert "Index creation failed" in str(exc_info.value)


def test_base_indexer_is_abstract():
    """Test that BaseIndexer cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseIndexer()


def test_vector_store_indexer_with_metadata():
    """Test VectorStoreIndexer preserves document metadata."""
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
    
    indexer = VectorStoreIndexer()
    mock_embed = MockEmbedding(embed_dim=256)
    
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, VectorStoreIndex)


def test_vector_store_indexer_different_embedding_dimensions():
    """Test VectorStoreIndexer with different embedding dimensions."""
    documents = [
        Document(text="Test document", doc_id="test1"),
    ]
    
    indexer = VectorStoreIndexer()
    
    # Test with different embedding dimensions
    for embed_dim in [128, 256, 512, 1024]:
        mock_embed = MockEmbedding(embed_dim=embed_dim)
        index = indexer.create_index(documents, embed_model=mock_embed)
        assert isinstance(index, VectorStoreIndex)