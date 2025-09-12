"""Tests for document indexing implementations."""

from unittest.mock import patch

import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from pydantic import ValidationError

from src.indexing import DocumentIndexer, PropertyGraphIndexer, VectorStoreIndexer
from src.indexing.property_graph import _select_extractors
from src.testing.mock_chain import MockLLMWithChain


def test__vector_store_indexer__creates_index():
    """Create vector index from three documents."""
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


def test__vector_store_indexer__custom_batch_size():
    """Custom batch size controls memory usage during indexing."""
    documents = [Document(text=f"Document {i}", doc_id=f"doc_{i}") for i in range(10)]

    # Smaller batch size for memory-constrained environments
    indexer = VectorStoreIndexer(insert_batch_size=512)
    assert indexer.insert_batch_size == 512

    mock_embed = MockEmbedding(embed_dim=256)
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, VectorStoreIndex)


def test__vector_store_indexer__empty_documents():
    """Empty document list creates valid empty index."""
    documents = []

    indexer = VectorStoreIndexer()
    mock_embed = MockEmbedding(embed_dim=256)

    # Should create valid index even with no documents
    index = indexer.create_index(documents, embed_model=mock_embed)
    assert isinstance(index, VectorStoreIndex)


def test__vector_store_indexer__error_handling():
    """Indexing errors are logged and re-raised."""
    documents = [Document(text="Test document", doc_id="test1")]
    indexer = VectorStoreIndexer()

    # Simulate index creation failure
    with patch.object(VectorStoreIndex, "from_documents", side_effect=Exception("Index creation failed")):
        with pytest.raises(Exception) as exc_info:
            indexer.create_index(documents)
        assert "Index creation failed" in str(exc_info.value)


def test__document_indexer__is_abstract():
    """DocumentIndexer cannot be instantiated directly."""
    with pytest.raises(TypeError):
        DocumentIndexer()


def test__vector_store_indexer__validates_input_types():
    """Pydantic validates documents must be a list."""
    indexer = VectorStoreIndexer()
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
    assert isinstance(index, VectorStoreIndex)

    # Valid: list with Documents should work
    docs = [Document(text="Test", doc_id="1")]
    index = indexer.create_index(docs, embed_model=mock_embed)
    assert isinstance(index, VectorStoreIndex)


# ----------------------------------------------
# PROPERTY GRAPH INDEXER TESTS
# ----------------------------------------------


def test__property_graph_indexer__creates_index():
    """Create property graph index from documents."""
    documents = [
        Document(text="Alice knows Bob", doc_id="1"),
        Document(text="Bob works at TechCorp", doc_id="2"),
        Document(text="TechCorp is in Silicon Valley", doc_id="3"),
    ]

    indexer = PropertyGraphIndexer(show_progress=False, embed_kg_nodes=False)
    mock_embed = MockEmbedding(embed_dim=256)

    index = indexer.create_index(documents, embed_model=mock_embed)

    from llama_index.core import PropertyGraphIndex

    assert isinstance(index, PropertyGraphIndex)
    assert hasattr(index, "property_graph_store")


def test__property_graph_indexer__with_llm():
    """LLM enables entity extraction for richer graphs."""
    documents = [
        Document(text="Apple Inc. was founded by Steve Jobs", doc_id="1"),
        Document(text="Microsoft was founded by Bill Gates", doc_id="2"),
    ]

    mock_llm = MockLLMWithChain(chain=["Entities: Apple Inc., Steve Jobs, founder relationship"])

    indexer = PropertyGraphIndexer(llm=mock_llm, show_progress=False, embed_kg_nodes=False)
    mock_embed = MockEmbedding(embed_dim=256)

    index = indexer.create_index(documents, embed_model=mock_embed)

    from llama_index.core import PropertyGraphIndex

    assert isinstance(index, PropertyGraphIndex)


def test__property_graph_indexer__validates_input_types():
    """Pydantic validates documents must be a list."""
    indexer = PropertyGraphIndexer(show_progress=False)
    mock_embed = MockEmbedding(embed_dim=256)

    # Invalid: string instead of list
    with pytest.raises(ValidationError):
        indexer.create_index("not a list", embed_model=mock_embed)

    # Valid: empty list
    index = indexer.create_index([], embed_model=mock_embed)
    from llama_index.core import PropertyGraphIndex

    assert isinstance(index, PropertyGraphIndex)


def test__select_extractors__helper():
    """Helper selects appropriate extractors based on configuration."""
    from llama_index.core.indices.property_graph.transformations import (
        ImplicitPathExtractor,
        SimpleLLMPathExtractor,
    )

    # No LLM: only implicit extractor
    extractors = _select_extractors(None, None)
    assert len(extractors) == 1
    assert isinstance(extractors[0], ImplicitPathExtractor)

    # With LLM: both extractors
    mock_llm = MockLLMWithChain(chain=["test"])
    extractors = _select_extractors(None, mock_llm)
    assert len(extractors) == 2
    assert isinstance(extractors[0], SimpleLLMPathExtractor)
    assert isinstance(extractors[1], ImplicitPathExtractor)

    # Custom extractors: use as-is
    custom = [ImplicitPathExtractor()]
    extractors = _select_extractors(custom, mock_llm)
    assert extractors == custom
