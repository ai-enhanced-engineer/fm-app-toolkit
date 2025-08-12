"""Tests for VectorStore indexing demonstrating core indexing concepts."""

from unittest.mock import patch

import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

from fm_app_toolkit.indexing import BaseIndexer, VectorStoreIndexer


def test_vector_store_indexer_creates_index():
    """Demonstrate basic index creation from documents.
    
    This test shows the fundamental operation of creating a searchable
    vector index from text documents using embeddings.
    """
    documents = [
        Document(text="First document about RAG pipelines", doc_id="1"),
        Document(text="Second document about embeddings", doc_id="2"),
        Document(text="Third document about retrieval", doc_id="3"),
    ]
    
    indexer = VectorStoreIndexer(show_progress=False)
    mock_embed = MockEmbedding(embed_dim=256)
    
    index = indexer.create_index(documents, embed_model=mock_embed)
    
    assert isinstance(index, VectorStoreIndex)
    # Index can create a retriever for searching
    assert index.as_retriever() is not None


def test_vector_store_indexer_custom_batch_size():
    """Demonstrate configuration of indexing parameters.
    
    Batch size affects memory usage and performance when indexing
    large document collections.
    """
    documents = [
        Document(text=f"Document {i}", doc_id=f"doc_{i}")
        for i in range(10)
    ]
    
    # Smaller batch size for memory-constrained environments
    indexer = VectorStoreIndexer(insert_batch_size=512)
    assert indexer.insert_batch_size == 512
    
    mock_embed = MockEmbedding(embed_dim=256)
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, VectorStoreIndex)


def test_vector_store_indexer_empty_documents():
    """Demonstrate edge case handling with empty document list.
    
    Indexers should handle edge cases gracefully without errors.
    """
    documents = []
    
    indexer = VectorStoreIndexer()
    mock_embed = MockEmbedding(embed_dim=256)
    
    # Should create valid index even with no documents
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, VectorStoreIndex)


def test_vector_store_indexer_error_handling():
    """Demonstrate proper error propagation in indexing pipeline.
    
    Errors during index creation should be logged and re-raised
    for proper handling by the application.
    """
    documents = [Document(text="Test document", doc_id="test1")]
    indexer = VectorStoreIndexer()
    
    # Simulate index creation failure
    with patch.object(
        VectorStoreIndex,
        'from_documents',
        side_effect=Exception("Index creation failed")
    ):
        with pytest.raises(Exception) as exc_info:
            indexer.create_index(documents)
        assert "Index creation failed" in str(exc_info.value)


def test_base_indexer_is_abstract():
    """Demonstrate that BaseIndexer enforces implementation of interface.
    
    Abstract base classes ensure consistent interface across different
    indexer implementations.
    """
    with pytest.raises(TypeError):
        BaseIndexer()