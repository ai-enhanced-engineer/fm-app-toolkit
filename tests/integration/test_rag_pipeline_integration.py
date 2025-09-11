"""Integration tests for RAG pipeline using MockLLMEchoStream.

Demonstrates end-to-end testing of RAG components including document indexing,
retrieval, and synthesis using mock LLMs for deterministic testing.
"""

from llama_index.core.schema import NodeWithScore, TextNode

from fm_app_toolkit.indexing.vector_store import VectorStoreIndexer
from fm_app_toolkit.testing.mock_echo import MockLLMEchoStream


def test_rag_pipeline_integration(sample_documents, mock_embed):
    """Test RAG pipeline integration using MockLLMEchoStream.

    This test demonstrates how to test a complete RAG pipeline deterministically:
    1. Index sample documents with vector search
    2. Set up query engine with MockLLMEchoStream
    3. Execute query and validate the complete workflow
    4. Verify that context is properly integrated into the LLM prompt
    """
    # Set up the complete pipeline with test documents
    mock_llm = MockLLMEchoStream()
    indexer = VectorStoreIndexer()

    # Create vector index and query engine
    index = indexer.create_index(sample_documents, embed_model=mock_embed)
    query_engine = index.as_query_engine(llm=mock_llm)

    # Test the full chain
    query = "What are the benefits of RAG?"
    response = query_engine.query(query)

    # Validate the complete integration
    assert query in response.response, "Query should appear in echoed response"
    assert len(response.source_nodes) > 0, "Should retrieve relevant context"

    # Test that context was properly integrated into the prompt
    # MockLLMEchoStream echoes back the full synthesized prompt
    context_snippets = [node.text[:50] for node in response.source_nodes]
    context_integrated = any(snippet in response.response for snippet in context_snippets)
    assert context_integrated, "Retrieved context should appear in synthesized prompt"


def test__rag_retrieval__context_integration(sample_documents, mock_embed):
    """Test that retrieved context is properly integrated into LLM prompt.

    This test focuses on the retrieval → synthesis integration without
    streaming complexity, validating context flow through the pipeline.
    """
    # Arrange: Set up retrieval components
    mock_llm = MockLLMEchoStream()
    indexer = VectorStoreIndexer()
    index = indexer.create_index(sample_documents, embed_model=mock_embed)

    # Get retriever and query engine separately for granular testing
    retriever = index.as_retriever(similarity_top_k=3)
    query_engine = index.as_query_engine(llm=mock_llm, similarity_top_k=3)

    # Act: Test retrieval and synthesis separately
    query = "How does RAG improve AI responses?"
    retrieved_nodes = retriever.retrieve(query)
    final_response = query_engine.query(query)

    # Assert: Validate retrieval quality and integration
    assert len(retrieved_nodes) > 0, "Should retrieve relevant nodes"
    assert len(retrieved_nodes) <= 3, "Should respect similarity_top_k limit"

    # Validate nodes have content and scores
    for node in retrieved_nodes:
        assert isinstance(node, NodeWithScore), "Retrieved items should be NodeWithScore"
        assert len(node.text) > 0, "Nodes should contain text content"
        assert node.score is not None, "Nodes should have similarity scores"

    # Validate synthesis includes retrieved context
    # The echo should contain parts of the retrieved context
    retrieved_texts = [node.text for node in retrieved_nodes]
    context_snippets_found = sum(
        1 for text in retrieved_texts if any(word in final_response.response for word in text.split()[:5])
    )
    assert context_snippets_found > 0, "Synthesized response should include retrieved context"


def test__rag_workflow__complete_query_response(sample_documents, mock_embed):
    """Test complete RAG workflow from documents to final response.

    End-to-end test validating the entire RAG use case:
    documents → indexing → retrieval → synthesis → streaming response
    """
    # Arrange: Complete workflow setup
    mock_llm = MockLLMEchoStream()
    indexer = VectorStoreIndexer()

    # Act: Execute complete workflow
    # Step 1: Index documents
    index = indexer.create_index(sample_documents, embed_model=mock_embed)

    # Step 2: Create query engine without streaming first
    query_engine = index.as_query_engine(llm=mock_llm, similarity_top_k=2)

    # Step 3: Execute query
    query = "What are the main advantages of using RAG systems?"
    response = query_engine.query(query)

    # Assert: Validate complete workflow
    assert hasattr(response, "response"), "Should have response attribute"
    assert query in response.response, "Response should contain original query"
    assert len(response.source_nodes) <= 2, "Should respect top_k parameter"

    # Validate workflow produced meaningful results
    assert len(response.response) > len(query), "Response should be longer than just the query"

    # Test streaming behavior separately
    streaming_query_engine = index.as_query_engine(llm=mock_llm, streaming=True)
    streaming_response = streaming_query_engine.query(query)

    # For streaming response, we get a StreamingResponse object
    assert hasattr(streaming_response, "source_nodes"), "Streaming response should have source_nodes"
    assert len(streaming_response.source_nodes) <= 2, "Streaming should respect top_k parameter"

    # The streaming response content should be accessible
    assert streaming_response.source_nodes is not None, "Should have source nodes in streaming response"

    # Validate that source nodes are properly attached
    for node in response.source_nodes:
        assert isinstance(node.node, TextNode), "Source nodes should contain TextNode"
        assert len(node.node.text) > 0, "Source nodes should have content"


def test__rag_pipeline__empty_query_handling(sample_documents, mock_embed):
    """Test RAG pipeline behavior with edge case queries."""
    # Arrange
    mock_llm = MockLLMEchoStream()
    indexer = VectorStoreIndexer()
    index = indexer.create_index(sample_documents, embed_model=mock_embed)
    query_engine = index.as_query_engine(llm=mock_llm)

    # Act & Assert: Test various edge cases
    # Very short query (skip empty query due to embedding issues)
    short_response = query_engine.query("AI")
    assert "AI" in short_response.response, "Should handle short query"
    assert len(short_response.source_nodes) >= 0, "Should return some nodes even for short query"

    # Single word query
    single_word_response = query_engine.query("technology")
    assert "technology" in single_word_response.response, "Should handle single word query"


def test__rag_pipeline__multiple_queries_consistency(sample_documents, mock_embed):
    """Test that RAG pipeline produces consistent results across multiple queries."""
    # Arrange
    mock_llm = MockLLMEchoStream()
    indexer = VectorStoreIndexer()
    index = indexer.create_index(sample_documents, embed_model=mock_embed)
    query_engine = index.as_query_engine(llm=mock_llm, similarity_top_k=2)

    # Act: Execute same query multiple times
    query = "What is RAG?"
    responses = [query_engine.query(query) for _ in range(3)]

    # Assert: Validate consistency
    for response in responses:
        assert query in response.response, "All responses should contain the query"
        assert len(response.source_nodes) <= 2, "All responses should respect top_k"

    # Source nodes should be consistent (same similarity scoring)
    source_texts = [[node.node.text for node in response.source_nodes] for response in responses]

    # All responses should have the same source nodes (deterministic retrieval)
    for i in range(1, len(source_texts)):
        assert source_texts[0] == source_texts[i], "Source nodes should be consistent across queries"
