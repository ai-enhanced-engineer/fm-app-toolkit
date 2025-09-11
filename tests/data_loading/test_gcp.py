"""Tests for fm_app_toolkit.data_loading.gcp module."""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document
from pydantic import ValidationError

from fm_app_toolkit.data_loading import GCPDocumentRepository
from fm_app_toolkit.data_loading.gcp import _parse_gcs_uri


# GCPDocumentRepository Tests
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


def test__gcp_repository__validates_location_type():
    """Pydantic validates location must be a string."""
    repo = GCPDocumentRepository()

    # Invalid: None instead of string
    with pytest.raises(ValidationError):
        repo.load_documents(location=None)

    # Invalid: dict instead of string
    with pytest.raises(ValidationError):
        repo.load_documents(location={"bucket": "test"})


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


# GCS URI Parser Tests
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
