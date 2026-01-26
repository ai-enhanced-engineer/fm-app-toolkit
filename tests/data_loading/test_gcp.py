"""Tests for src.data_loading.gcp module."""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document
from pydantic import ValidationError

from src.data_loading import DataLoadingError, GCPDocumentRepository, GCSError, GCSLoadError, GCSURIError
from src.data_loading.gcp import _parse_gcs_uri


# GCPDocumentRepository Tests
@patch("src.data_loading.gcp.GCSReader")
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


@patch("src.data_loading.gcp.GCSReader")
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


@patch("src.data_loading.gcp.GCSReader")
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
    with pytest.raises(GCSURIError, match="URI must start with gs://"):
        repo.load_documents(location="s3://bucket/file.txt")

    # Invalid: missing gs:// prefix
    with pytest.raises(GCSURIError, match="URI must start with gs://"):
        repo.load_documents(location="bucket/file.txt")


@patch("src.data_loading.gcp.GCSReader")
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
    """Invalid URI format raises GCSURIError."""
    with pytest.raises(GCSURIError, match="URI must start with gs://"):
        _parse_gcs_uri("s3://bucket/file.txt")

    with pytest.raises(GCSURIError, match="URI must start with gs://"):
        _parse_gcs_uri("http://bucket/file.txt")

    with pytest.raises(GCSURIError, match="URI must start with gs://"):
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


# GCS Exception Hierarchy Tests
class TestGCSExceptionHierarchy:
    """Tests for GCS exception classes."""

    def test__gcs_uri_error__preserves_context(self) -> None:
        """GCSURIError stores URI and reason."""
        error = GCSURIError("s3://bucket/file.txt", "URI must start with gs://")

        assert error.uri == "s3://bucket/file.txt"
        assert error.reason == "URI must start with gs://"
        assert "s3://bucket/file.txt" in str(error)
        assert "URI must start with gs://" in str(error)

    def test__gcs_uri_error__formats_message(self) -> None:
        """GCSURIError message follows consistent format."""
        error = GCSURIError("http://example.com", "Invalid protocol")

        expected = "Invalid GCS URI 'http://example.com': Invalid protocol"
        assert str(error) == expected

    def test__gcs_load_error__preserves_context(self) -> None:
        """GCSLoadError stores location and reason."""
        error = GCSLoadError("gs://bucket/file.txt", "Permission denied")

        assert error.location == "gs://bucket/file.txt"
        assert error.reason == "Permission denied"
        assert "gs://bucket/file.txt" in str(error)
        assert "Permission denied" in str(error)

    def test__gcs_load_error__formats_message(self) -> None:
        """GCSLoadError message follows consistent format."""
        error = GCSLoadError("gs://my-bucket/data/", "Network timeout")

        expected = "Failed to load from 'gs://my-bucket/data/': Network timeout"
        assert str(error) == expected

    def test__gcs_exception_hierarchy__gcs_uri_error_inheritance(self) -> None:
        """GCSURIError inherits from GCSError and DataLoadingError."""
        error = GCSURIError("test://uri", "test reason")

        assert isinstance(error, GCSError)
        assert isinstance(error, DataLoadingError)
        assert isinstance(error, Exception)

    def test__gcs_exception_hierarchy__gcs_load_error_inheritance(self) -> None:
        """GCSLoadError inherits from GCSError and DataLoadingError."""
        error = GCSLoadError("gs://test", "test error")

        assert isinstance(error, GCSError)
        assert isinstance(error, DataLoadingError)
        assert isinstance(error, Exception)

    def test__gcs_exception_hierarchy__base_classes_instantiable(self) -> None:
        """Base exception classes can be instantiated."""
        data_loading_error = DataLoadingError("Generic data loading error")
        gcs_error = GCSError("Generic GCS error")

        assert str(data_loading_error) == "Generic data loading error"
        assert str(gcs_error) == "Generic GCS error"
        assert isinstance(gcs_error, DataLoadingError)


@patch("src.data_loading.gcp.GCSReader")
def test__gcp_document_repository__raises_gcs_load_error_on_os_error(mock_gcs_reader):
    """GCS load failures from OSError raise GCSLoadError."""
    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.side_effect = OSError("Permission denied")
    mock_gcs_reader.return_value = mock_reader_instance

    repo = GCPDocumentRepository()

    with pytest.raises(GCSLoadError) as exc_info:
        repo.load_documents(location="gs://test-bucket/file.txt")

    assert exc_info.value.location == "gs://test-bucket/file.txt"
    assert "Permission denied" in exc_info.value.reason


@patch("src.data_loading.gcp.GCSReader")
def test__gcp_document_repository__raises_gcs_load_error_on_io_error(mock_gcs_reader):
    """GCS load failures from IOError raise GCSLoadError."""
    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.side_effect = IOError("Network unreachable")
    mock_gcs_reader.return_value = mock_reader_instance

    repo = GCPDocumentRepository()

    with pytest.raises(GCSLoadError) as exc_info:
        repo.load_documents(location="gs://test-bucket/data/")

    assert exc_info.value.location == "gs://test-bucket/data/"
    assert "Network unreachable" in exc_info.value.reason


@patch("src.data_loading.gcp.GCSReader")
def test__gcp_document_repository__raises_gcs_load_error_on_permission_error(mock_gcs_reader):
    """GCS load failures from PermissionError raise GCSLoadError."""
    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.side_effect = PermissionError("Access denied")
    mock_gcs_reader.return_value = mock_reader_instance

    repo = GCPDocumentRepository()

    with pytest.raises(GCSLoadError) as exc_info:
        repo.load_documents(location="gs://secure-bucket/file.txt")

    assert exc_info.value.location == "gs://secure-bucket/file.txt"
    assert "Access denied" in exc_info.value.reason
