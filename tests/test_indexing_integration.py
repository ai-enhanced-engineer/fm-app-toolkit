"""Integration tests demonstrating the full document indexing pipeline."""


from fm_app_toolkit.data_loading import LocalDocumentRepository
from fm_app_toolkit.indexing import PropertyGraphIndexer, VectorStoreIndexer
from fm_app_toolkit.testing.mocks import MockLLMWithChain

# All fixtures (test_data_dir, sample_documents, mock_embed) come from conftest.py


def test_full_indexing_pipeline(test_data_dir, mock_embed):
    """Demonstrate complete pipeline: Load documents → Create indexes.
    
    This integration test shows how documents flow from data loading
    through to index creation, preserving metadata throughout.
    """
    # Step 1: Load documents from filesystem
    repo = LocalDocumentRepository(
        input_dir=str(test_data_dir),
        required_exts=[".txt"]
    )
    documents = repo.load_documents()
    assert len(documents) == 3, "Should load all test documents"
    
    # Verify metadata is preserved from loading
    for doc in documents:
        assert doc.metadata.get("file_name") is not None
        assert doc.metadata.get("file_path") is not None
    
    # Step 2: Create vector index for similarity search
    vector_indexer = VectorStoreIndexer(show_progress=False)
    vector_index = vector_indexer.create_index(documents, embed_model=mock_embed)
    assert vector_index is not None
    assert vector_index.as_retriever() is not None
    
    # Step 3: Create property graph index for relationship queries
    graph_indexer = PropertyGraphIndexer(
        show_progress=False,
        embed_kg_nodes=False
    )
    graph_index = graph_indexer.create_index(documents, embed_model=mock_embed)
    assert graph_index is not None


def test_technical_content_preservation(sample_documents, mock_embed):
    """Verify that technical content and terminology is preserved during indexing.
    
    Real-world documents contain domain-specific terms that must be
    accurately preserved in the index for effective retrieval.
    """
    indexer = VectorStoreIndexer(show_progress=False)
    index = indexer.create_index(sample_documents, embed_model=mock_embed)
    
    # Technical terms from our test documents
    expected_terms = ["RAG", "chunking", "embeddings", "retrieval", "guardrails"]
    
    # Verify terms are preserved in the index's document store
    docstore = index.storage_context.docstore
    all_text = ""
    for doc_id in docstore.docs:
        node = docstore.get_node(doc_id)
        if node and hasattr(node, 'text'):
            all_text += node.text.lower()
    
    for term in expected_terms:
        assert term.lower() in all_text, f"Technical term '{term}' should be preserved"


def test_vector_vs_graph_indexing_comparison(sample_documents, mock_embed):
    """Compare VectorStore and PropertyGraph indexing approaches.
    
    This test demonstrates that both indexing strategies can process
    the same documents but create different representations:
    - VectorStore: Optimized for similarity search via embeddings
    - PropertyGraph: Optimized for relationship and entity queries
    """
    # Same documents, different indexing strategies
    vector_indexer = VectorStoreIndexer(show_progress=False)
    graph_indexer = PropertyGraphIndexer(
        show_progress=False,
        embed_kg_nodes=False
    )
    
    # Create both types of indexes
    vector_index = vector_indexer.create_index(sample_documents, embed_model=mock_embed)
    graph_index = graph_indexer.create_index(sample_documents, embed_model=mock_embed)
    
    # Both indexes are valid and serve different purposes
    assert vector_index is not None
    assert graph_index is not None
    
    # Vector index supports retriever creation for similarity search
    assert vector_index.as_retriever() is not None
    
    # Graph index has property graph store for relationship queries
    assert hasattr(graph_index, 'property_graph_store')


def test_property_graph_with_llm_extraction(sample_documents, mock_embed):
    """Demonstrate enhanced entity extraction when LLM is available.
    
    With an LLM, PropertyGraphIndexer can extract more meaningful
    entities and relationships from technical documents.
    """
    # Mock LLM simulates extraction of technical entities
    mock_llm = MockLLMWithChain(chain=[
        "Entities: RAG, Pipeline, Vector Database",
        "Entities: Evaluation, Metrics, Guardrails",
        "Entities: Context, Chunking, Retrieval"
    ])
    
    indexer = PropertyGraphIndexer(
        llm=mock_llm,
        show_progress=False,
        embed_kg_nodes=False
    )
    
    index = indexer.create_index(sample_documents, embed_model=mock_embed)
    
    # Verify property graph was created with extracted entities
    assert index is not None
    assert hasattr(index, 'property_graph_store')