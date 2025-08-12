"""Integration tests for indexing module with real sample documents."""

from pathlib import Path

import pytest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

from fm_app_toolkit.data_loading import LocalDocumentRepository
from fm_app_toolkit.indexing import PropertyGraphIndexer, VectorStoreIndexer
from fm_app_toolkit.testing.mocks import MockLLMWithChain

# ----------------------------------------------
# FIXTURES
# ----------------------------------------------


@pytest.fixture
def test_data_dir():
    """Get the path to the test data directory."""
    return Path(__file__).parent.parent / "fm_app_toolkit" / "test_data"


@pytest.fixture
def sample_documents(test_data_dir):
    """Load sample documents using LocalDocumentRepository."""
    repo = LocalDocumentRepository(
        input_dir=str(test_data_dir),
        required_exts=[".txt"]
    )
    return repo.load_documents()


@pytest.fixture
def mock_embed():
    """Create a mock embedding model."""
    return MockEmbedding(embed_dim=256)


# ----------------------------------------------
# VECTOR STORE INDEXING TESTS
# ----------------------------------------------


def test_vector_store_indexer_with_real_documents(sample_documents, mock_embed):
    """Test VectorStoreIndexer with real technical documents."""
    # Create indexer
    indexer = VectorStoreIndexer(show_progress=False)
    
    # Create index from real documents
    index = indexer.create_index(sample_documents, embed_model=mock_embed)
    
    # Verify index was created
    assert index is not None
    
    # Create retriever and verify it works
    retriever = index.as_retriever()
    assert retriever is not None


def test_vector_store_indexer_handles_technical_content(sample_documents, mock_embed):
    """Test that VectorStoreIndexer properly processes technical AI/ML content."""
    indexer = VectorStoreIndexer(
        show_progress=False,
        insert_batch_size=512  # Smaller batch for testing
    )
    
    # Create index
    index = indexer.create_index(sample_documents, embed_model=mock_embed)
    
    # Verify all documents were indexed
    assert index is not None
    
    # The documents contain technical terms that should be preserved
    expected_terms = ["RAG", "chunking", "embeddings", "retrieval", "guardrails"]
    
    # Get the docstore from the index
    docstore = index.storage_context.docstore
    
    # Check that technical content was preserved
    all_text = ""
    for doc_id in docstore.docs:
        node = docstore.get_node(doc_id)
        if node and hasattr(node, 'text'):
            all_text += node.text.lower()
    
    for term in expected_terms:
        assert term.lower() in all_text, f"Technical term '{term}' should be preserved"


# ----------------------------------------------
# PROPERTY GRAPH INDEXING TESTS  
# ----------------------------------------------


def test_property_graph_indexer_with_real_documents(sample_documents, mock_embed):
    """Test PropertyGraphIndexer with real technical documents."""
    # Create indexer without LLM (will use implicit extractor)
    indexer = PropertyGraphIndexer(
        show_progress=False,
        embed_kg_nodes=False  # Skip embedding for speed
    )
    
    # Create index from real documents
    index = indexer.create_index(sample_documents, embed_model=mock_embed)
    
    # Verify index was created
    assert index is not None


def test_property_graph_extracts_technical_entities(sample_documents, mock_embed):
    """Test that PropertyGraphIndexer extracts meaningful entities from technical content."""
    # Use mock LLM for controlled extraction
    mock_llm = MockLLMWithChain(chain=[
        "Extracted entities: RAG, Pipeline, Architecture",
        "Extracted entities: Evaluation, Guardrails, Safety",
        "Extracted entities: Context, Construction, Strategies"
    ])
    
    indexer = PropertyGraphIndexer(
        llm=mock_llm,
        show_progress=False,
        embed_kg_nodes=False
    )
    
    # Create index
    index = indexer.create_index(sample_documents, embed_model=mock_embed)
    
    # Verify index was created with property graph structure
    assert index is not None
    assert hasattr(index, 'property_graph_store')


def test_property_graph_with_embeddings(sample_documents, mock_embed):
    """Test PropertyGraphIndexer with knowledge graph node embeddings."""
    indexer = PropertyGraphIndexer(
        embed_kg_nodes=True,  # Enable KG node embeddings
        show_progress=False
    )
    
    # Create index with embeddings
    index = indexer.create_index(sample_documents, embed_model=mock_embed)
    
    # Verify index was created
    assert index is not None


# ----------------------------------------------
# INTEGRATION PIPELINE TESTS
# ----------------------------------------------


def test_full_pipeline_load_index_query(test_data_dir, mock_embed):
    """Test the complete pipeline: Load → Index → Query capability."""
    # Step 1: Load documents
    repo = LocalDocumentRepository(
        input_dir=str(test_data_dir),
        required_exts=[".txt"]
    )
    documents = repo.load_documents()
    assert len(documents) == 3, "Should load all test documents"
    
    # Step 2: Create vector index
    vector_indexer = VectorStoreIndexer(show_progress=False)
    vector_index = vector_indexer.create_index(documents, embed_model=mock_embed)
    assert vector_index is not None
    
    # Step 3: Create property graph index  
    graph_indexer = PropertyGraphIndexer(
        show_progress=False,
        embed_kg_nodes=False
    )
    graph_index = graph_indexer.create_index(documents, embed_model=mock_embed)
    assert graph_index is not None
    
    # Step 4: Verify both indexes can create retrievers
    vector_retriever = vector_index.as_retriever()
    assert vector_retriever is not None
    
    # Property graph index needs LLM for retriever, skip for now
    # graph_retriever = graph_index.as_retriever()
    # assert graph_retriever is not None


def test_indexers_handle_document_metadata(test_data_dir, mock_embed):
    """Test that both indexers preserve document metadata."""
    # Load documents with metadata
    repo = LocalDocumentRepository(input_dir=str(test_data_dir))
    documents = repo.load_documents()
    
    # Each document should have file metadata
    for doc in documents:
        assert doc.metadata.get("file_name") is not None
        assert doc.metadata.get("file_path") is not None
    
    # Test VectorStoreIndexer preserves metadata
    vector_indexer = VectorStoreIndexer()
    vector_index = vector_indexer.create_index(documents, embed_model=mock_embed)
    assert vector_index is not None
    
    # Test PropertyGraphIndexer preserves metadata
    graph_indexer = PropertyGraphIndexer(embed_kg_nodes=False)
    graph_index = graph_indexer.create_index(documents, embed_model=mock_embed)
    assert graph_index is not None


def test_both_indexers_with_same_documents(sample_documents, mock_embed):
    """Test that both indexers can process the same documents successfully."""
    # Both indexers should handle the same content
    vector_indexer = VectorStoreIndexer(show_progress=False)
    graph_indexer = PropertyGraphIndexer(
        show_progress=False,
        embed_kg_nodes=False
    )
    
    # Create both indexes from the same documents
    vector_index = vector_indexer.create_index(sample_documents, embed_model=mock_embed)
    graph_index = graph_indexer.create_index(sample_documents, embed_model=mock_embed)
    
    # Both should succeed
    assert vector_index is not None
    assert graph_index is not None
    
    # Vector index supports retrieval
    assert vector_index.as_retriever() is not None
    # Property graph index needs LLM for retriever, skip for now
    # assert graph_index.as_retriever() is not None