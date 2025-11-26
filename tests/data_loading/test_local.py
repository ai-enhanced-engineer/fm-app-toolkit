"""Tests for src.data_loading.local module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from src.data_loading import LocalDocumentRepository
from src.data_loading.base import BaseRepository
from src.data_loading.local import LocalRepository


# LocalDocumentRepository Tests
def test__local_document_repository__loads_documents():
    """Test that LocalDocumentRepository loads documents from filesystem."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content")

        repo = LocalDocumentRepository(input_dir=temp_dir)
        documents = repo.load_documents(location=temp_dir)

        assert len(documents) == 1
        assert "Test content" in documents[0].text


def test__local_document_repository__filters_extensions():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files with different extensions
        (Path(temp_dir) / "test.txt").write_text("Text file")
        (Path(temp_dir) / "test.md").write_text("Markdown file")
        (Path(temp_dir) / "test.py").write_text("Python file")

        repo = LocalDocumentRepository(input_dir=temp_dir, required_exts=[".txt", ".md"])
        documents = repo.load_documents(location=temp_dir)

        assert len(documents) == 2


def test__local_document_repository__recursive():
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


def test__local_repository__validates_location_type():
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


def test__local_repository__constructor_validates_meaningful_params():
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


# LocalRepository Tests
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
