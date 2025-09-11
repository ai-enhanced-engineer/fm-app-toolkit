"""Integration tests demonstrating RAG pipeline construction with real documents."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

from fm_app_toolkit.data_loading import LocalDocumentRepository
from fm_app_toolkit.data_loading.local import LocalRepository
from fm_app_toolkit.testing.mocks import MockLLMWithChain

# ----------------------------------------------
# FIXTURES
# ----------------------------------------------


@pytest.fixture
def samples_dir():
    """Path to test documents: RAG architecture, context construction, and guardrails."""
    return Path(__file__).parent.parent / "fm_app_toolkit" / "test_data"


@pytest.fixture
def sample_repository(samples_dir):
    """Repository configured to load .txt files from test_data directory."""
    return LocalDocumentRepository(input_dir=str(samples_dir), required_exts=[".txt"])


# ----------------------------------------------
# DOCUMENT LOADING TESTS
# ----------------------------------------------


def test__load_sample__documents(sample_repository, samples_dir):
    """Load three GenAI documents and verify metadata extraction."""
    documents = sample_repository.load_documents(location=str(samples_dir))

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


def test__document_metadata__extraction(sample_repository, samples_dir):
    """Each document has file_name, file_path, and file_size metadata."""
    documents = sample_repository.load_documents(location=str(samples_dir))

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


def test__build_simple__rag_pipeline(sample_repository, samples_dir):
    """Build a complete RAG pipeline: load docs → create index → query."""
    # Load documents
    documents = sample_repository.load_documents(location=str(samples_dir))

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


def test__document_chunking__simulation(sample_repository, samples_dir):
    """Split documents into paragraph chunks for granular retrieval."""
    documents = sample_repository.load_documents(location=str(samples_dir))

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
# SEARCH AND RETRIEVAL TESTS
# ----------------------------------------------


def test__content_search__simulation(sample_repository, samples_dir):
    """Keyword search ranks documents by term frequency."""
    documents = sample_repository.load_documents(location=str(samples_dir))

    # Simulate a simple keyword search
    def search_documents(query: str, documents: list[Document]) -> list[tuple[Document, int]]:
        """Count query occurrences in each document."""
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


# ----------------------------------------------
# CSV DATA REPOSITORY INTEGRATION TESTS
# ----------------------------------------------


@pytest.fixture
def sample_csv_data():
    """Generate realistic sample data for CSV testing."""
    return pd.DataFrame(
        {
            "customer_id": range(1, 101),
            "name": [f"Customer_{i:03d}" for i in range(1, 101)],
            "email": [f"customer{i}@example.com" for i in range(1, 101)],
            "age": [20 + (i % 60) for i in range(1, 101)],  # Ages 20-79
            "signup_date": pd.date_range("2023-01-01", periods=100, freq="D"),
            "revenue": [round(100 + (i * 15.5), 2) for i in range(1, 101)],
            "active": [i % 3 != 0 for i in range(1, 101)],  # Mix of True/False
        }
    )


def test__local_repository__real_world_csv_workflow(sample_csv_data):
    """End-to-end workflow: create CSV, load with LocalRepository, analyze data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create realistic CSV file
        csv_file = Path(temp_dir) / "customers.csv"
        sample_csv_data.to_csv(csv_file, index=False)

        # Load using LocalRepository
        repo = LocalRepository()
        df = repo.load_data(str(csv_file))

        # Verify loaded data matches original
        assert len(df) == 100
        assert list(df.columns) == ["customer_id", "name", "email", "age", "signup_date", "revenue", "active"]

        # Perform realistic data analysis operations
        # 1. Filter active customers
        active_customers = df[df["active"]]
        assert len(active_customers) > 0

        # 2. Calculate revenue statistics
        total_revenue = df["revenue"].sum()
        avg_revenue = df["revenue"].mean()
        assert total_revenue > 0
        assert avg_revenue > 0

        # 3. Age demographics
        young_customers = df[df["age"] < 30]
        senior_customers = df[df["age"] >= 60]
        assert len(young_customers) + len(senior_customers) < len(df)  # Some middle-aged customers

        # 4. Date range analysis
        df["signup_date"] = pd.to_datetime(df["signup_date"])
        oldest_signup = df["signup_date"].min()
        newest_signup = df["signup_date"].max()
        assert oldest_signup < newest_signup


def test__local_repository__multiple_csv_files_workflow():
    """Load and combine data from multiple CSV files in a workflow."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = LocalRepository()

        # Create multiple related CSV files
        customers_data = pd.DataFrame(
            {"customer_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "city": ["NYC", "LA", "Chicago"]}
        )

        orders_data = pd.DataFrame(
            {
                "order_id": [101, 102, 103, 104],
                "customer_id": [1, 2, 1, 3],
                "amount": [150.00, 250.00, 99.99, 300.00],
                "status": ["completed", "pending", "completed", "shipped"],
            }
        )

        # Save to separate files
        customers_file = Path(temp_dir) / "customers.csv"
        orders_file = Path(temp_dir) / "orders.csv"
        customers_data.to_csv(customers_file, index=False)
        orders_data.to_csv(orders_file, index=False)

        # Load each file
        customers_df = repo.load_data(str(customers_file))
        orders_df = repo.load_data(str(orders_file))

        # Verify separate loading
        assert len(customers_df) == 3
        assert len(orders_df) == 4
        assert "customer_id" in customers_df.columns
        assert "customer_id" in orders_df.columns

        # Simulate join operation (typical business workflow)
        combined = orders_df.merge(customers_df, on="customer_id", how="left")
        assert len(combined) == 4
        assert "name" in combined.columns
        assert "amount" in combined.columns

        # Verify join worked correctly
        alice_orders = combined[combined["name"] == "Alice"]
        assert len(alice_orders) == 2  # Alice has 2 orders


def test__local_repository__csv_data_quality_checks():
    """Integration test for data quality validation after loading."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV with various data quality issues by writing directly
        csv_content = """id,score,category,timestamp,flag
1,95.5,A,2023-01-01,True
2,87.2,B,invalid-date,False
,92.0,A,2023-01-03,True
4,,C,2023-01-04,maybe
5,88.8,,2023-01-05,False"""

        csv_file = Path(temp_dir) / "quality_issues.csv"
        csv_file.write_text(csv_content)

        repo = LocalRepository()
        df = repo.load_data(str(csv_file))

        # Verify data loaded despite issues
        assert len(df) == 5

        # Perform data quality checks (common real-world workflow)
        # 1. Check for missing values
        missing_counts = df.isnull().sum()
        assert missing_counts["id"] > 0  # Should detect missing ID
        assert missing_counts["score"] > 0  # Should detect missing score

        # 2. Check for empty strings (pandas may convert empty strings to NaN)
        empty_or_null = ((df == "") | df.isnull()).sum()
        assert empty_or_null["category"] > 0  # Should detect empty or null category

        # 3. Data type validation
        numeric_columns = ["id", "score"]
        for col in numeric_columns:
            # pandas should handle type conversion automatically
            assert col in df.columns

        # 4. Date parsing simulation
        try:
            df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
            invalid_dates = df["timestamp_parsed"].isnull().sum()
            assert invalid_dates > 0  # Should detect invalid date
        except Exception:
            pass  # Expected for invalid date formats


def test__local_repository__performance_with_realistic_data():
    """Test LocalRepository performance with larger, more realistic dataset."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate larger dataset (5000 rows)
        large_data = pd.DataFrame(
            {
                "transaction_id": range(1, 5001),
                "customer_id": [f"CUST_{i:06d}" for i in range(1, 5001)],
                "product_category": ["Electronics", "Clothing", "Books", "Home"] * 1250,
                "amount": [round(10 + (i * 0.75), 2) for i in range(1, 5001)],
                "transaction_date": pd.date_range("2023-01-01", periods=5000, freq="H"),
                "payment_method": ["Credit", "Debit", "PayPal", "Cash"] * 1250,
                "discount_applied": [i % 7 == 0 for i in range(1, 5001)],  # ~14% get discounts
                "notes": [f"Transaction notes for order {i}" for i in range(1, 5001)],
            }
        )

        csv_file = Path(temp_dir) / "large_transactions.csv"
        large_data.to_csv(csv_file, index=False)

        repo = LocalRepository()

        # Time the loading (basic performance check)
        import time

        start_time = time.time()
        df = repo.load_data(str(csv_file))
        load_time = time.time() - start_time

        # Verify correct loading
        assert len(df) == 5000
        assert len(df.columns) == 8

        # Performance assertion (should load 5000 rows quickly)
        assert load_time < 5.0, f"Loading took {load_time:.2f}s, should be under 5s"

        # Verify data integrity after loading
        assert df["transaction_id"].nunique() == 5000  # All IDs unique
        assert df["amount"].sum() > 0  # Positive total
        assert df["discount_applied"].sum() > 0  # Some discounts applied


@pytest.mark.integration
def test__local_repository__csv_to_document_conversion():
    """Integration test: CSV data → Document format for RAG pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create knowledge base CSV
        kb_data = pd.DataFrame(
            {
                "doc_id": ["DOC001", "DOC002", "DOC003"],
                "title": ["RAG Architecture", "Embeddings Guide", "Vector Stores"],
                "content": [
                    "RAG combines retrieval and generation for better accuracy...",
                    "Embeddings convert text into dense vector representations...",
                    "Vector stores enable efficient similarity search...",
                ],
                "category": ["Architecture", "ML", "Infrastructure"],
                "last_updated": ["2023-01-15", "2023-02-20", "2023-03-10"],
            }
        )

        csv_file = Path(temp_dir) / "knowledge_base.csv"
        kb_data.to_csv(csv_file, index=False)

        # Load CSV data
        repo = LocalRepository()
        df = repo.load_data(str(csv_file))

        # Convert to LlamaIndex Document format (common integration pattern)
        documents = []
        for _, row in df.iterrows():
            doc = Document(
                text=f"Title: {row['title']}\n\nContent: {row['content']}",
                metadata={
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "category": row["category"],
                    "last_updated": row["last_updated"],
                },
            )
            documents.append(doc)

        # Verify conversion
        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)

        # Check document content and metadata
        rag_doc = next(doc for doc in documents if "RAG" in doc.text)
        assert rag_doc.metadata["category"] == "Architecture"
        assert "retrieval and generation" in rag_doc.text

        # Simulate building an index from the converted documents
        Settings.embed_model = MockEmbedding(embed_dim=256)
        Settings.llm = MockLLMWithChain(
            chain=["Based on the knowledge base, RAG improves accuracy by combining retrieval with generation."]
        )

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        # Test querying the CSV-derived knowledge base
        response = query_engine.query("What is RAG architecture?")
        assert len(response.response) > 0
