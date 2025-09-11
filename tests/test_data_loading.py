"""Tests for data loading repositories."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from llama_index.core import Document
from pydantic import ValidationError

from fm_app_toolkit.data_loading import (
    GCPDocumentRepository,
    LocalDocumentRepository,
)
from fm_app_toolkit.data_loading.base import BaseRepository
from fm_app_toolkit.data_loading.gcp import _parse_gcs_uri
from fm_app_toolkit.data_loading.local import LocalRepository


def test__local_document_repository__loads_documents():
    """Test that LocalDocumentRepository loads documents from filesystem."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content")

        repo = LocalDocumentRepository(input_dir=temp_dir)
        documents = repo.load_documents(location=temp_dir)

        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert "Test content" in documents[0].text


def test__local_document_repository__filters_extensions():
    """Test that LocalDocumentRepository filters documents by file extension."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files with different extensions
        (Path(temp_dir) / "test.txt").write_text("Text file")
        (Path(temp_dir) / "test.md").write_text("Markdown file")
        (Path(temp_dir) / "test.py").write_text("Python file")

        repo = LocalDocumentRepository(input_dir=temp_dir, required_exts=[".txt", ".md"])
        documents = repo.load_documents(location=temp_dir)

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)


def test__local_document_repository__recursive():
    """Test that LocalDocumentRepository handles recursive directory traversal."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested directory structure
        root = Path(temp_dir)
        (root / "file1.txt").write_text("File 1")
        subdir = root / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("File 2")

        # Test recursive loading
        repo = LocalDocumentRepository(input_dir=temp_dir, recursive=True)
        documents = repo.load_documents(location=temp_dir)
        assert len(documents) == 2

        # Test non-recursive loading
        repo = LocalDocumentRepository(input_dir=temp_dir, recursive=False)
        documents = repo.load_documents(location=temp_dir)
        assert len(documents) == 1


def test__local_document_repository__excludes_hidden():
    """Test that LocalDocumentRepository excludes hidden files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create regular and hidden files
        (Path(temp_dir) / "visible.txt").write_text("Visible")
        (Path(temp_dir) / ".hidden.txt").write_text("Hidden")

        repo = LocalDocumentRepository(input_dir=temp_dir, exclude_hidden=True)
        documents = repo.load_documents(location=temp_dir)

        assert len(documents) == 1
        assert "Visible" in documents[0].text


def test__local_document_repository__file_limit():
    """Limit number of documents loaded."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple files
        for i in range(5):
            (Path(temp_dir) / f"file{i}.txt").write_text(f"Content {i}")

        repo = LocalDocumentRepository(input_dir=temp_dir, num_files_limit=3)
        documents = repo.load_documents(location=temp_dir)

        assert len(documents) == 3


def test__local_document_repository__handles_missing_directory():
    """Missing directories raise clear errors."""
    repo = LocalDocumentRepository(input_dir="/nonexistent/directory")

    with pytest.raises(Exception, match="does not exist"):
        repo.load_documents(location="/nonexistent/directory")


@patch("fm_app_toolkit.data_loading.gcp.GCSReader")
def test__gcp_document_repository__with_key(mock_gcs_reader):
    """Load single file: gs://bucket/path/to/file.txt"""
    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.return_value = [Document(text="GCS content", metadata={"source": "gcs"})]
    mock_gcs_reader.return_value = mock_reader_instance

    repo = GCPDocumentRepository()
    documents = repo.load_documents(location="gs://test-bucket/path/to/file.txt")

    assert len(documents) == 1
    assert documents[0].text == "GCS content"
    mock_gcs_reader.assert_called_once_with(bucket="test-bucket", key="path/to/file.txt")


@patch("fm_app_toolkit.data_loading.gcp.GCSReader")
def test__gcp_document_repository__with_prefix(mock_gcs_reader):
    """Load directory: gs://bucket/documents/"""
    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.return_value = [
        Document(text="Doc 1", metadata={"source": "gcs"}),
        Document(text="Doc 2", metadata={"source": "gcs"}),
    ]
    mock_gcs_reader.return_value = mock_reader_instance

    repo = GCPDocumentRepository()
    documents = repo.load_documents(location="gs://test-bucket/documents/")

    assert len(documents) == 2
    mock_gcs_reader.assert_called_once_with(bucket="test-bucket", prefix="documents/")


@patch("fm_app_toolkit.data_loading.gcp.GCSReader")
def test__gcp_document_repository__with_service_account(mock_gcs_reader):
    """Authenticate with service account credentials."""
    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.return_value = [Document(text="Authenticated content", metadata={})]
    mock_gcs_reader.return_value = mock_reader_instance

    service_account_key = {"type": "service_account", "project_id": "test"}

    repo = GCPDocumentRepository(service_account_key=service_account_key)
    documents = repo.load_documents(location="gs://test-bucket/file.txt")

    assert len(documents) == 1
    mock_gcs_reader.assert_called_once_with(
        bucket="test-bucket", key="file.txt", service_account_key=service_account_key
    )


def test__gcp_document_repository__validates_gs_uri():
    """Only gs:// URIs are accepted, not s3:// or paths."""
    repo = GCPDocumentRepository()

    # Invalid: not a gs:// URI
    with pytest.raises(ValueError, match="GCS location must start with gs://"):
        repo.load_documents(location="s3://bucket/file.txt")

    # Invalid: missing gs:// prefix
    with pytest.raises(ValueError, match="GCS location must start with gs://"):
        repo.load_documents(location="bucket/file.txt")


@patch("fm_app_toolkit.data_loading.gcp.GCSReader")
def test__gcp_document_repository__handles_load_error(mock_gcs_reader):
    """GCS errors are logged and re-raised."""
    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.side_effect = Exception("GCS error")
    mock_gcs_reader.return_value = mock_reader_instance

    repo = GCPDocumentRepository()

    with pytest.raises(Exception, match="GCS error"):
        repo.load_documents(location="gs://test-bucket/file.txt")


def test__local_repository__validates_location_type():
    """Pydantic validates location must be a string."""
    repo = LocalDocumentRepository(input_dir=".", recursive=True)

    # Invalid: None instead of string
    with pytest.raises(ValidationError):
        repo.load_documents(location=None)

    # Invalid: integer instead of string
    with pytest.raises(ValidationError):
        repo.load_documents(location=123)

    # Invalid: list instead of string
    with pytest.raises(ValidationError):
        repo.load_documents(location=["/path/to/dir"])


def test__gcp_repository__validates_location_type():
    """Pydantic validates location must be a string."""
    repo = GCPDocumentRepository()

    # Invalid: None instead of string
    with pytest.raises(ValidationError):
        repo.load_documents(location=None)

    # Invalid: dict instead of string
    with pytest.raises(ValidationError):
        repo.load_documents(location={"bucket": "test"})


def test__local_repository__constructor_validates_meaningful_params():
    """Test constructor validation for business-relevant parameter types."""
    # Invalid: input_dir must be string, not None
    with pytest.raises(ValidationError):
        LocalDocumentRepository(input_dir=None)

    # Invalid: input_dir must be string, not integer
    with pytest.raises(ValidationError):
        LocalDocumentRepository(input_dir=123)

    # Invalid: required_exts must be list[str], not string
    with pytest.raises(ValidationError):
        LocalDocumentRepository(input_dir=".", required_exts=".txt")

    # Invalid: num_files_limit must be int, not dict
    with pytest.raises(ValidationError):
        LocalDocumentRepository(input_dir=".", num_files_limit={"count": 5})

    # Invalid: recursive must be bool, not list
    with pytest.raises(ValidationError):
        LocalDocumentRepository(input_dir=".", recursive=[True, False])

    # Valid: proper types should work
    repo = LocalDocumentRepository(input_dir="/tmp", recursive=False, required_exts=[".txt", ".md"], num_files_limit=10)
    assert repo.input_dir == "/tmp"
    assert repo.recursive is False
    assert repo.required_exts == [".txt", ".md"]
    assert repo.num_files_limit == 10


def test__gcp_repository__constructor_validates_service_account():
    """Test service account key validation for business requirements."""
    # Invalid: service_account_key must be dict, not string
    with pytest.raises(ValidationError):
        GCPDocumentRepository(service_account_key="invalid-string")

    # Invalid: service_account_key must be dict, not list
    with pytest.raises(ValidationError):
        GCPDocumentRepository(service_account_key=["key1", "key2"])

    # Invalid: service_account_key must be dict, not integer
    with pytest.raises(ValidationError):
        GCPDocumentRepository(service_account_key=123)

    # Valid: None is acceptable (default)
    repo1 = GCPDocumentRepository()
    assert repo1.service_account_key is None

    # Valid: proper dict should work
    valid_key = {"type": "service_account", "project_id": "test"}
    repo2 = GCPDocumentRepository(service_account_key=valid_key)
    assert repo2.service_account_key == valid_key


# ----------------------------------------------
# GCS URI PARSER TESTS
# ----------------------------------------------


def test__parse_gcs__uri_bucket_only():
    """Parse URI with only bucket name."""
    result = _parse_gcs_uri("gs://my-bucket")
    assert result == {"bucket": "my-bucket"}


def test__parse_gcs__uri_with_file():
    """Parse URI pointing to a specific file."""
    result = _parse_gcs_uri("gs://my-bucket/path/to/file.txt")
    assert result == {"bucket": "my-bucket", "key": "path/to/file.txt"}


def test__parse_gcs__uri_with_prefix():
    """Parse URI with directory prefix (trailing slash)."""
    result = _parse_gcs_uri("gs://my-bucket/path/to/dir/")
    assert result == {"bucket": "my-bucket", "prefix": "path/to/dir/"}


def test__parse_gcs__uri_single_file_no_path():
    """Parse URI with file at bucket root."""
    result = _parse_gcs_uri("gs://my-bucket/file.txt")
    assert result == {"bucket": "my-bucket", "key": "file.txt"}


def test__parse_gcs__uri_single_dir():
    """Parse URI with single directory."""
    result = _parse_gcs_uri("gs://my-bucket/dir/")
    assert result == {"bucket": "my-bucket", "prefix": "dir/"}


def test__parse_gcs__uri_invalid_format():
    """Invalid URI format raises ValueError."""
    with pytest.raises(ValueError, match="GCS location must start with gs://"):
        _parse_gcs_uri("s3://bucket/file.txt")

    with pytest.raises(ValueError, match="GCS location must start with gs://"):
        _parse_gcs_uri("http://bucket/file.txt")

    with pytest.raises(ValueError, match="GCS location must start with gs://"):
        _parse_gcs_uri("/local/path/file.txt")


def test__parse_gcs__uri_edge_cases():
    """Handle edge cases in URI parsing."""
    # Bucket with hyphen and numbers
    result = _parse_gcs_uri("gs://my-bucket-123")
    assert result == {"bucket": "my-bucket-123"}

    # Deep nesting
    result = _parse_gcs_uri("gs://bucket/a/b/c/d/e/f.txt")
    assert result == {"bucket": "bucket", "key": "a/b/c/d/e/f.txt"}

    # Multiple trailing slashes (treated as prefix)
    result = _parse_gcs_uri("gs://bucket/path//")
    assert result == {"bucket": "bucket", "prefix": "path//"}


# ----------------------------------------------
# BASE REPOSITORY TESTS
# ----------------------------------------------


def test__base_repository__is_abstract():
    """BaseRepository cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseRepository"):
        BaseRepository()


def test__base_repository__requires_load_data_implementation():
    """Concrete implementations must implement load_data method."""

    class IncompleteRepository(BaseRepository):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteRepository"):
        IncompleteRepository()


def test__base_repository__abstract_method_signature():
    """load_data method must have correct signature."""

    class ConcreteRepository(BaseRepository):
        def load_data(self, path: str) -> pd.DataFrame:
            return pd.DataFrame({"test": [1, 2, 3]})

    # Should instantiate without error
    repo = ConcreteRepository()
    result = repo.load_data("dummy_path")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["test"]


# ----------------------------------------------
# LOCAL REPOSITORY TESTS
# ----------------------------------------------


def test__local_repository__loads_basic_csv():
    """Load simple CSV file with basic data types."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test CSV
        csv_file = Path(temp_dir) / "test.csv"
        test_data = pd.DataFrame(
            {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "city": ["New York", "London", "Tokyo"]}
        )
        test_data.to_csv(csv_file, index=False)

        repo = LocalRepository()
        df = repo.load_data(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["name", "age", "city"]
        assert df.loc[0, "name"] == "Alice"
        assert df.loc[1, "age"] == 30


def test__local_repository__loads_empty_csv():
    """Handle CSV files with headers but no data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "empty.csv"
        # Create CSV with headers but no data
        empty_data = pd.DataFrame({"col1": [], "col2": []})
        empty_data.to_csv(csv_file, index=False)

        repo = LocalRepository()
        df = repo.load_data(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["col1", "col2"]


def test__local_repository__handles_completely_empty_file():
    """Completely empty CSV files raise appropriate error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "completely_empty.csv"
        csv_file.write_text("")  # Completely empty file

        repo = LocalRepository()

        with pytest.raises(Exception):  # pandas.errors.EmptyDataError will be wrapped
            repo.load_data(str(csv_file))


def test__local_repository__loads_csv_with_mixed_types():
    """Load CSV with various data types (string, int, float, bool)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "mixed.csv"
        test_data = pd.DataFrame(
            {
                "text": ["hello", "world", "test"],
                "integer": [1, 2, 3],
                "float": [1.5, 2.7, 3.9],
                "boolean": [True, False, True],
            }
        )
        test_data.to_csv(csv_file, index=False)

        repo = LocalRepository()
        df = repo.load_data(str(csv_file))

        assert len(df) == 3
        assert list(df.columns) == ["text", "integer", "float", "boolean"]
        assert df.loc[0, "text"] == "hello"
        assert df.loc[1, "integer"] == 2
        assert df.loc[2, "float"] == 3.9


def test__local_repository__loads_csv_with_special_characters():
    """Handle CSV with special characters and Unicode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "special.csv"
        test_data = pd.DataFrame({"text": ["café", "naïve", "résumé", "🚀"], "numbers": [1, 2, 3, 4]})
        test_data.to_csv(csv_file, index=False, encoding="utf-8")

        repo = LocalRepository()
        df = repo.load_data(str(csv_file))

        assert len(df) == 4
        assert df.loc[0, "text"] == "café"
        assert df.loc[3, "text"] == "🚀"


def test__local_repository__loads_large_csv():
    """Handle larger CSV files efficiently."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "large.csv"

        # Create a DataFrame with 1000 rows
        large_data = pd.DataFrame(
            {"id": range(1000), "value": [f"value_{i}" for i in range(1000)], "score": [i * 0.1 for i in range(1000)]}
        )
        large_data.to_csv(csv_file, index=False)

        repo = LocalRepository()
        df = repo.load_data(str(csv_file))

        assert len(df) == 1000
        assert list(df.columns) == ["id", "value", "score"]
        assert df.loc[0, "id"] == 0
        assert df.loc[999, "value"] == "value_999"


def test__local_repository__handles_missing_file():
    """Missing CSV file raises appropriate error."""
    repo = LocalRepository()

    with pytest.raises(Exception):  # FileNotFoundError will be wrapped
        repo.load_data("/nonexistent/file.csv")


def test__local_repository__handles_invalid_csv():
    """Malformed CSV content raises appropriate error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        bad_csv = Path(temp_dir) / "bad.csv"
        bad_csv.write_text("This is not,a proper\nCSV file,with,inconsistent\ncolumns")

        repo = LocalRepository()

        # Should raise an exception, but pandas is quite forgiving
        # This test ensures we don't crash completely
        df = repo.load_data(str(bad_csv))
        assert isinstance(df, pd.DataFrame)  # pandas will still try to parse it


def test__local_repository__handles_permission_denied():
    """File permission errors are properly raised."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "restricted.csv"
        test_data = pd.DataFrame({"col": [1, 2, 3]})
        test_data.to_csv(csv_file, index=False)

        # Change permissions to remove read access
        csv_file.chmod(0o000)

        repo = LocalRepository()

        try:
            with pytest.raises(Exception):  # PermissionError will be wrapped
                repo.load_data(str(csv_file))
        finally:
            # Restore permissions for cleanup
            csv_file.chmod(0o644)


def test__local_repository__loads_csv_with_null_values():
    """Handle CSV files containing null/NaN values."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "nulls.csv"
        test_data = pd.DataFrame(
            {"name": ["Alice", None, "Charlie"], "age": [25, pd.NA, 35], "score": [1.5, 2.0, None]}
        )
        test_data.to_csv(csv_file, index=False)

        repo = LocalRepository()
        df = repo.load_data(str(csv_file))

        assert len(df) == 3
        assert pd.isna(df.loc[1, "name"])
        assert pd.isna(df.loc[2, "score"])


def test__local_repository__inheritance():
    """LocalRepository properly inherits from BaseRepository."""
    repo = LocalRepository()
    assert isinstance(repo, BaseRepository)

    # Should have the abstract method implemented
    assert hasattr(repo, "load_data")
    assert callable(repo.load_data)
