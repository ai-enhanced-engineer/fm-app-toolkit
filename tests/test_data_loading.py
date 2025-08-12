"""Tests for data loading repositories."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document
from pydantic import ValidationError

from fm_app_toolkit.data_loading import (
    GCPDocumentRepository,
    LocalDocumentRepository,
)
from fm_app_toolkit.data_loading.gcp import _parse_gcs_uri


def test_local_document_repository_loads_documents():
    """Basic document loading from filesystem."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content")

        repo = LocalDocumentRepository(input_dir=temp_dir)
        documents = repo.load_documents(location=temp_dir)

        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert "Test content" in documents[0].text


def test_local_document_repository_filters_extensions():
    """Filter documents by file extension (.txt, .md, etc)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files with different extensions
        (Path(temp_dir) / "test.txt").write_text("Text file")
        (Path(temp_dir) / "test.md").write_text("Markdown file")
        (Path(temp_dir) / "test.py").write_text("Python file")

        repo = LocalDocumentRepository(input_dir=temp_dir, required_exts=[".txt", ".md"])
        documents = repo.load_documents(location=temp_dir)

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)


def test_local_document_repository_recursive():
    """Recursive vs non-recursive directory traversal."""
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


def test_local_document_repository_excludes_hidden():
    """Hidden files (starting with .) are excluded."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create regular and hidden files
        (Path(temp_dir) / "visible.txt").write_text("Visible")
        (Path(temp_dir) / ".hidden.txt").write_text("Hidden")

        repo = LocalDocumentRepository(input_dir=temp_dir, exclude_hidden=True)
        documents = repo.load_documents(location=temp_dir)

        assert len(documents) == 1
        assert "Visible" in documents[0].text


def test_local_document_repository_file_limit():
    """Limit number of documents loaded."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple files
        for i in range(5):
            (Path(temp_dir) / f"file{i}.txt").write_text(f"Content {i}")

        repo = LocalDocumentRepository(input_dir=temp_dir, num_files_limit=3)
        documents = repo.load_documents(location=temp_dir)

        assert len(documents) == 3


def test_local_document_repository_handles_missing_directory():
    """Missing directories raise clear errors."""
    repo = LocalDocumentRepository(input_dir="/nonexistent/directory")

    with pytest.raises(Exception, match="does not exist"):
        repo.load_documents(location="/nonexistent/directory")


@patch("fm_app_toolkit.data_loading.gcp.GCSReader")
def test_gcp_document_repository_with_key(mock_gcs_reader):
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
def test_gcp_document_repository_with_prefix(mock_gcs_reader):
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
def test_gcp_document_repository_with_service_account(mock_gcs_reader):
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


def test_gcp_document_repository_validates_gs_uri():
    """Only gs:// URIs are accepted, not s3:// or paths."""
    repo = GCPDocumentRepository()
    
    # Invalid: not a gs:// URI
    with pytest.raises(ValueError, match="GCS location must start with gs://"):
        repo.load_documents(location="s3://bucket/file.txt")
    
    # Invalid: missing gs:// prefix
    with pytest.raises(ValueError, match="GCS location must start with gs://"):
        repo.load_documents(location="bucket/file.txt")


@patch("fm_app_toolkit.data_loading.gcp.GCSReader")
def test_gcp_document_repository_handles_load_error(mock_gcs_reader):
    """GCS errors are logged and re-raised."""
    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.side_effect = Exception("GCS error")
    mock_gcs_reader.return_value = mock_reader_instance

    repo = GCPDocumentRepository()

    with pytest.raises(Exception, match="GCS error"):
        repo.load_documents(location="gs://test-bucket/file.txt")


def test_local_repository_validates_location_type():
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


def test_gcp_repository_validates_location_type():
    """Pydantic validates location must be a string."""
    repo = GCPDocumentRepository()
    
    # Invalid: None instead of string
    with pytest.raises(ValidationError):
        repo.load_documents(location=None)
    
    # Invalid: dict instead of string  
    with pytest.raises(ValidationError):
        repo.load_documents(location={"bucket": "test"})


# ----------------------------------------------
# GCS URI PARSER TESTS
# ----------------------------------------------


def test_parse_gcs_uri_bucket_only():
    """Parse URI with only bucket name."""
    result = _parse_gcs_uri("gs://my-bucket")
    assert result == {"bucket": "my-bucket"}


def test_parse_gcs_uri_with_file():
    """Parse URI pointing to a specific file."""
    result = _parse_gcs_uri("gs://my-bucket/path/to/file.txt")
    assert result == {"bucket": "my-bucket", "key": "path/to/file.txt"}


def test_parse_gcs_uri_with_prefix():
    """Parse URI with directory prefix (trailing slash)."""
    result = _parse_gcs_uri("gs://my-bucket/path/to/dir/")
    assert result == {"bucket": "my-bucket", "prefix": "path/to/dir/"}


def test_parse_gcs_uri_single_file_no_path():
    """Parse URI with file at bucket root."""
    result = _parse_gcs_uri("gs://my-bucket/file.txt")
    assert result == {"bucket": "my-bucket", "key": "file.txt"}


def test_parse_gcs_uri_single_dir():
    """Parse URI with single directory."""
    result = _parse_gcs_uri("gs://my-bucket/dir/")
    assert result == {"bucket": "my-bucket", "prefix": "dir/"}


def test_parse_gcs_uri_invalid_format():
    """Invalid URI format raises ValueError."""
    with pytest.raises(ValueError, match="GCS location must start with gs://"):
        _parse_gcs_uri("s3://bucket/file.txt")
    
    with pytest.raises(ValueError, match="GCS location must start with gs://"):
        _parse_gcs_uri("http://bucket/file.txt")
    
    with pytest.raises(ValueError, match="GCS location must start with gs://"):
        _parse_gcs_uri("/local/path/file.txt")


def test_parse_gcs_uri_edge_cases():
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
