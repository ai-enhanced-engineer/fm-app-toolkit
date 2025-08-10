"""Tests for data loading repositories."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document

from fm_app_toolkit.data_loading import (
    GCPDocumentRepository,
    LocalDocumentRepository,
)


def test_local_document_repository_loads_documents():
    """Test LocalDocumentRepository loads documents from directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content")
        
        repo = LocalDocumentRepository(input_dir=temp_dir)
        documents = repo.load_documents()
        
        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert "Test content" in documents[0].text


def test_local_document_repository_filters_extensions():
    """Test LocalDocumentRepository filters by file extension."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files with different extensions
        (Path(temp_dir) / "test.txt").write_text("Text file")
        (Path(temp_dir) / "test.md").write_text("Markdown file")
        (Path(temp_dir) / "test.py").write_text("Python file")
        
        repo = LocalDocumentRepository(
            input_dir=temp_dir,
            required_exts=[".txt", ".md"]
        )
        documents = repo.load_documents()
        
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)


def test_local_document_repository_recursive():
    """Test LocalDocumentRepository loads documents recursively."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested directory structure
        root = Path(temp_dir)
        (root / "file1.txt").write_text("File 1")
        subdir = root / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("File 2")
        
        # Test recursive loading
        repo = LocalDocumentRepository(input_dir=temp_dir, recursive=True)
        documents = repo.load_documents()
        assert len(documents) == 2
        
        # Test non-recursive loading
        repo = LocalDocumentRepository(input_dir=temp_dir, recursive=False)
        documents = repo.load_documents()
        assert len(documents) == 1


def test_local_document_repository_excludes_hidden():
    """Test LocalDocumentRepository excludes hidden files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create regular and hidden files
        (Path(temp_dir) / "visible.txt").write_text("Visible")
        (Path(temp_dir) / ".hidden.txt").write_text("Hidden")
        
        repo = LocalDocumentRepository(
            input_dir=temp_dir,
            exclude_hidden=True
        )
        documents = repo.load_documents()
        
        assert len(documents) == 1
        assert "Visible" in documents[0].text


def test_local_document_repository_file_limit():
    """Test LocalDocumentRepository respects file limit."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple files
        for i in range(5):
            (Path(temp_dir) / f"file{i}.txt").write_text(f"Content {i}")
        
        repo = LocalDocumentRepository(
            input_dir=temp_dir,
            num_files_limit=3
        )
        documents = repo.load_documents()
        
        assert len(documents) == 3


def test_local_document_repository_handles_missing_directory():
    """Test LocalDocumentRepository handles missing directory gracefully."""
    repo = LocalDocumentRepository(input_dir="/nonexistent/directory")
    
    with pytest.raises(Exception):
        repo.load_documents()


@patch("fm_app_toolkit.data_loading.gcp.GCSReader")
def test_gcp_document_repository_with_key(mock_gcs_reader):
    """Test GCPDocumentRepository loads single document by key."""
    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.return_value = [
        Document(text="GCS content", metadata={"source": "gcs"})
    ]
    mock_gcs_reader.return_value = mock_reader_instance
    
    repo = GCPDocumentRepository(
        bucket="test-bucket",
        key="path/to/file.txt"
    )
    documents = repo.load_documents()
    
    assert len(documents) == 1
    assert documents[0].text == "GCS content"
    mock_gcs_reader.assert_called_once_with(
        bucket="test-bucket",
        key="path/to/file.txt"
    )


@patch("fm_app_toolkit.data_loading.gcp.GCSReader")
def test_gcp_document_repository_with_prefix(mock_gcs_reader):
    """Test GCPDocumentRepository loads multiple documents by prefix."""
    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.return_value = [
        Document(text="Doc 1", metadata={"source": "gcs"}),
        Document(text="Doc 2", metadata={"source": "gcs"}),
    ]
    mock_gcs_reader.return_value = mock_reader_instance
    
    repo = GCPDocumentRepository(
        bucket="test-bucket",
        prefix="documents/"
    )
    documents = repo.load_documents()
    
    assert len(documents) == 2
    mock_gcs_reader.assert_called_once_with(
        bucket="test-bucket",
        prefix="documents/"
    )


@patch("fm_app_toolkit.data_loading.gcp.GCSReader")
def test_gcp_document_repository_with_service_account(mock_gcs_reader):
    """Test GCPDocumentRepository uses service account key."""
    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.return_value = [
        Document(text="Authenticated content", metadata={})
    ]
    mock_gcs_reader.return_value = mock_reader_instance
    
    service_account_key = {"type": "service_account", "project_id": "test"}
    
    repo = GCPDocumentRepository(
        bucket="test-bucket",
        key="file.txt",
        service_account_key=service_account_key
    )
    documents = repo.load_documents()
    
    assert len(documents) == 1
    mock_gcs_reader.assert_called_once_with(
        bucket="test-bucket",
        key="file.txt",
        service_account_key=service_account_key
    )


def test_gcp_document_repository_requires_key_or_prefix():
    """Test GCPDocumentRepository raises error without key or prefix."""
    with pytest.raises(ValueError, match="Either 'key' or 'prefix' must be provided"):
        GCPDocumentRepository(bucket="test-bucket")


@patch("fm_app_toolkit.data_loading.gcp.GCSReader")
def test_gcp_document_repository_handles_load_error(mock_gcs_reader):
    """Test GCPDocumentRepository handles loading errors gracefully."""
    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.side_effect = Exception("GCS error")
    mock_gcs_reader.return_value = mock_reader_instance
    
    repo = GCPDocumentRepository(
        bucket="test-bucket",
        key="file.txt"
    )
    
    with pytest.raises(Exception, match="GCS error"):
        repo.load_documents()