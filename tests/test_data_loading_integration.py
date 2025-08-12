"""Integration tests for data loading with sample documents."""

from pathlib import Path

import pytest
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

from fm_app_toolkit.data_loading import LocalDocumentRepository
from fm_app_toolkit.testing.mocks import MockLLMWithChain

# ----------------------------------------------
# FIXTURES
# ----------------------------------------------


@pytest.fixture
def samples_dir():
    """Get the path to the test data directory."""
    return Path(__file__).parent.parent / "fm_app_toolkit" / "test_data"


@pytest.fixture
def sample_repository(samples_dir):
    """Create a repository for the sample documents."""
    return LocalDocumentRepository(input_dir=str(samples_dir), required_exts=[".txt"])


# ----------------------------------------------
# DOCUMENT LOADING TESTS
# ----------------------------------------------


def test_load_sample_documents(sample_repository):
    """Test loading sample text documents from samples directory."""
    documents = sample_repository.load_documents()

    # Verify all documents loaded
    assert len(documents) == 3, "Should load all three sample documents"

    # Check each document has substantial content
    for doc in documents:
        assert len(doc.text) > 500, f"Document {doc.metadata.get('file_name')} should have substantial content"
        assert doc.metadata.get("file_name") is not None
        assert doc.metadata.get("file_path") is not None

    # Verify we have documents about each topic
    file_names = [doc.metadata.get("file_name") for doc in documents]
    assert "rag_pipeline_architecture.txt" in file_names
    assert "context_construction_strategies.txt" in file_names
    assert "genai_evaluation_guardrails.txt" in file_names


def test_sample_content_coverage(sample_repository):
    """Verify sample documents cover key GenAI concepts."""
    documents = sample_repository.load_documents()
    all_text = " ".join([doc.text.lower() for doc in documents])

    # Key concepts that should be covered
    essential_concepts = [
        "retrieval-augmented generation",
        "rag",
        "embedding",
        "vector",
        "context construction",
        "guardrails",
        "evaluation",
        "chunking",
        "prompt engineering",
        "hallucination",
        "observability",
        "hybrid search",
        "llm-as-judge",
        "personalization",
    ]

    for concept in essential_concepts:
        assert concept.lower() in all_text, f"Missing essential concept: {concept}"


def test_document_metadata_extraction(sample_repository):
    """Test that metadata is properly extracted from documents."""
    documents = sample_repository.load_documents()

    for doc in documents:
        metadata = doc.metadata

        # Check required metadata fields
        assert "file_name" in metadata
        assert "file_path" in metadata
        assert "file_size" in metadata

        # Verify metadata values are reasonable
        assert metadata["file_name"].endswith(".txt")
        assert metadata["file_size"] > 0


# ----------------------------------------------
# RAG PIPELINE TESTS
# ----------------------------------------------


def test_build_simple_rag_pipeline(sample_repository):
    """Demonstrate building a simple RAG pipeline with sample docs."""
    # Load documents
    documents = sample_repository.load_documents()

    # Use mock LLM to avoid API calls
    mock_llm = MockLLMWithChain(
        chain=[
            "Based on the context, RAG (Retrieval-Augmented Generation) combines retrieval of relevant documents with generation to improve accuracy and reduce hallucinations.",
            "Context construction involves four levels: Basic RAG, Query Understanding, Memory Management, and Personalization.",
            "Guardrails include input validation for security and output checks for quality and safety.",
        ]
    )

    # Configure settings with mock LLM and embedding
    Settings.llm = mock_llm
    Settings.embed_model = MockEmbedding(embed_dim=256)
    Settings.chunk_size = 512

    # Build index
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # Test queries - the mock LLM will return predefined responses
    response = query_engine.query("What is RAG?")
    assert "retrieval" in response.response.lower()

    # Additional simple validation - just check we get responses
    response = query_engine.query("What are the levels of context construction?")
    assert len(response.response) > 0

    response = query_engine.query("What are guardrails in GenAI?")
    assert len(response.response) > 0


def test_document_chunking_simulation(sample_repository):
    """Test document chunking and retrieval patterns."""
    documents = sample_repository.load_documents()

    # Simulate chunking by paragraphs
    all_chunks = []
    for doc in documents:
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]

        for i, para in enumerate(paragraphs):
            if len(para) > 50:  # Skip very short chunks
                chunk = {"text": para, "source": doc.metadata.get("file_name"), "chunk_id": i, "length": len(para)}
                all_chunks.append(chunk)

    # Verify chunking produced reasonable results
    assert len(all_chunks) > len(documents) * 5, "Should produce multiple chunks per document"
    assert all(chunk["text"] for chunk in all_chunks), "All chunks should have text"
    assert all(chunk["length"] > 50 for chunk in all_chunks), "All chunks should meet minimum length"

    # Check chunk distribution across documents
    sources = set(chunk["source"] for chunk in all_chunks)
    assert len(sources) == 3, "Chunks should come from all three documents"


# ----------------------------------------------
# FILTERING AND LOADING TESTS
# ----------------------------------------------


def test_filtered_document_loading(samples_dir):
    """Test loading documents with specific filters."""
    # Load only one document with file limit
    repo = LocalDocumentRepository(input_dir=str(samples_dir), required_exts=[".txt"], num_files_limit=1)

    documents = repo.load_documents()
    assert len(documents) == 1, "Should respect file limit"


def test_recursive_loading_with_samples(samples_dir):
    """Test that recursive loading works with the test_data directory."""
    repo = LocalDocumentRepository(
        input_dir=str(samples_dir),  # Start from test_data directory
        recursive=True,
        required_exts=[".txt"],
    )

    documents = repo.load_documents()

    # Should find the text files in the test_data directory
    assert len(documents) == 3, "Should find exactly three test documents"

    # Verify the samples were included
    sample_files = {
        "rag_pipeline_architecture.txt",
        "context_construction_strategies.txt",
        "genai_evaluation_guardrails.txt",
    }
    loaded_files = {Path(doc.metadata.get("file_name")).name for doc in documents}
    assert sample_files.issubset(loaded_files), "Should include all sample files"


# ----------------------------------------------
# SEARCH AND RETRIEVAL TESTS
# ----------------------------------------------


def test_content_search_simulation(sample_repository):
    """Simulate searching for specific content in loaded documents."""
    documents = sample_repository.load_documents()

    # Simulate a simple keyword search
    def search_documents(query: str, documents: list[Document]) -> list[tuple[Document, int]]:
        """Simple keyword search returning documents and match counts."""
        results = []
        query_lower = query.lower()
        for doc in documents:
            count = doc.text.lower().count(query_lower)
            if count > 0:
                results.append((doc, count))
        return sorted(results, key=lambda x: x[1], reverse=True)

    # Search for RAG-related content
    rag_results = search_documents("retrieval-augmented generation", documents)
    assert len(rag_results) > 0, "Should find documents mentioning RAG"
    # The document with most mentions should be one of the RAG-focused docs
    top_doc = rag_results[0][0].metadata.get("file_name")
    assert top_doc in ["rag_pipeline_architecture.txt", "context_construction_strategies.txt"]

    # Search for guardrails content
    guardrails_results = search_documents("guardrails", documents)
    assert len(guardrails_results) > 0, "Should find documents mentioning guardrails"

    # Search for context construction
    context_results = search_documents("context construction", documents)
    assert len(context_results) > 0, "Should find documents about context construction"
