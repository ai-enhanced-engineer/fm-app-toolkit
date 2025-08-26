# Data Loading Module - Repository Pattern Implementation

## Overview

This module implements the **Repository Pattern** for document loading in FM (Foundation Model) applications. The Repository pattern provides a simple abstraction over data access, allowing us to swap between different data sources (local filesystem, cloud storage) without changing our application code.

> 📚 This implementation follows the principles described in [Chapter 2: Repository Pattern](https://www.cosmicpython.com/book/chapter_02_repository.html) from the book "Architecture Patterns with Python" by Harry Percival and Bob Gregory.

## Why the Repository Pattern?

The Repository pattern provides several key benefits:

1. **Testing without External Services**: Load test data from local files instead of requiring internet connectivity or cloud credentials
2. **Abstraction**: Hide the complexity of data access behind a simple interface
3. **Flexibility**: Easily swap between different data sources (local, GCS, S3, etc.)
4. **Testability**: Use mock or in-memory repositories for unit tests
5. **Separation of Concerns**: Keep business logic separate from infrastructure details

As described in [Cosmic Python Chapter 2](https://www.cosmicpython.com/book/chapter_02_repository.html#_what_is_the_repository_pattern):
> "The Repository pattern is a simplifying abstraction over data storage, allowing us to decouple our model layer from the data layer."

## 🚀 Quick Start

Want to see the Repository pattern in action immediately?

```bash
# See document loading and chunking working
make process-documents
```

This runs our `process_documents()` function which demonstrates:
- Loading documents from local filesystem using `LocalDocumentRepository`
- Chunking text with LlamaIndex's `SentenceSplitter`  
- Structured logging throughout the process
- Clean separation between CLI output and processing logs

## Testing Without External Services

One of the most valuable aspects of this pattern is enabling **offline development and testing**. Instead of requiring:
- Internet connectivity
- Cloud service credentials
- Actual cloud storage buckets
- API rate limits and costs

We can use `LocalDocumentRepository` to:
- Load test documents from local directories
- Run tests in CI/CD without external dependencies
- Develop features offline
- Create predictable, reproducible tests

Example of swapping repositories for testing:

```python
# In production
from fm_app_toolkit.data_loading import GCPDocumentRepository

def create_production_repository():
    return GCPDocumentRepository()

# In tests  
from fm_app_toolkit.data_loading import LocalDocumentRepository

def create_test_repository():
    return LocalDocumentRepository(
        input_dir="./tests/fixtures/documents"
    )

# Your application code doesn't change
def process_documents(repository: DocumentRepository, location: str):
    documents = repository.load_documents(location=location)
    # Process documents...

# Usage
prod_repo = create_production_repository()
process_documents(prod_repo, "gs://production-docs/documents/")

test_repo = create_test_repository()  
process_documents(test_repo, "./tests/fixtures/documents")
```

## Available Repositories

### BaseRepository (Abstract Base)

The abstract base class for generic data loading (CSV files, structured data):

```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseRepository(ABC):
    @abstractmethod
    def load_data(self, path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        raise NotImplementedError
```

**LocalRepository** - Concrete implementation for loading CSV data:
```python
from fm_app_toolkit.data_loading.local import LocalRepository

repo = LocalRepository()
df = repo.load_data("./data/customers.csv")
```

### DocumentRepository (Abstract Base)

The abstract base class that defines the interface all repositories must implement:

```python
from abc import ABC, abstractmethod
from llama_index.core import Document

class DocumentRepository(ABC):
    @abstractmethod
    def load_documents(self) -> list[Document]:
        """Load all documents from the repository."""
        raise NotImplementedError
```

This follows the **Dependency Inversion Principle** - our application depends on the abstraction, not concrete implementations.

### LocalDocumentRepository

Loads documents from the local filesystem using LlamaIndex's `SimpleDirectoryReader`.

**How it works:**
- Scans directories for documents
- Supports multiple file formats (.txt, .md, .pdf, .json, etc.)
- Can recursively search subdirectories
- Filters by file extensions

**Configuration:**
```python
from fm_app_toolkit.data_loading import LocalDocumentRepository

repo = LocalDocumentRepository(
    input_dir="./data",           # Directory to scan
    recursive=True,                # Search subdirectories
    required_exts=[".txt", ".md"], # Only load specific file types
    exclude_hidden=True,           # Skip hidden files
    num_files_limit=100           # Limit number of files
)

documents = repo.load_documents(location="./data")
```

**Use Cases:**
- Local development with sample data
- Unit and integration testing
- Processing documents from mounted volumes
- Batch processing of local files

### GCPDocumentRepository

Loads documents from Google Cloud Storage using LlamaIndex's `GCSReader`.

**How it works:**
- Connects to GCS buckets
- Can load single files or multiple files by prefix
- Handles authentication via service account keys or default credentials
- Returns documents with GCS metadata

**Configuration:**
```python
from fm_app_toolkit.data_loading import GCPDocumentRepository

# Load from GCS with optional service account key
repo = GCPDocumentRepository(
    service_account_key={...}  # Optional: explicit credentials
)

# Load documents using gs:// URI format
documents = repo.load_documents(location="gs://my-bucket/documents/")
```

**Use Cases:**
- Production deployments
- Processing cloud-stored documents
- Integrating with data pipelines
- Scalable document storage

## Usage Examples

### Basic Usage

```python
from fm_app_toolkit.data_loading import (
    DocumentRepository,
    LocalDocumentRepository,
    GCPDocumentRepository
)

# The beauty of the pattern - your function doesn't care about the source
def build_index(repository: DocumentRepository, location: str):
    """Build a search index from documents."""
    documents = repository.load_documents(location=location)
    
    # Process documents with LlamaIndex
    from llama_index.core import VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents)
    return index

# In development/testing
local_repo = LocalDocumentRepository(input_dir="./test_data")
dev_index = build_index(local_repo, "./test_data")

# In production
gcp_repo = GCPDocumentRepository()
prod_index = build_index(gcp_repo, "gs://prod-docs/knowledge/")
```

### Dependency Injection

```python
from typing import Protocol

class DocumentService:
    def __init__(self, repository: DocumentRepository):
        self.repository = repository
    
    def process_documents(self, location: str):
        documents = self.repository.load_documents(location=location)
        # Business logic here
        return processed_results

# Inject different repositories based on environment
import os

def create_repository() -> DocumentRepository:
    if os.getenv("ENVIRONMENT") == "production":
        return GCPDocumentRepository()
    else:
        return LocalDocumentRepository(
            input_dir=os.getenv("LOCAL_DATA_DIR", "./data")
        )

def get_data_location() -> str:
    if os.getenv("ENVIRONMENT") == "production":
        return f"gs://{os.getenv('GCS_BUCKET')}/{os.getenv('GCS_PREFIX', '')}"
    else:
        return os.getenv("LOCAL_DATA_DIR", "./data")

# Application bootstrap
repository = create_repository()
service = DocumentService(repository)
location = get_data_location()
results = service.process_documents(location)
```

## Testing Strategies

### Unit Testing with Local Repository

```python
import tempfile
from pathlib import Path
import pytest

def test_document_processing():
    # Create test documents in temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup test data
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content")
        
        # Use local repository for testing
        repo = LocalDocumentRepository(input_dir=temp_dir)
        documents = repo.load_documents(location=temp_dir)
        
        assert len(documents) == 1
        assert "Test content" in documents[0].text
```

### Mocking GCP Repository

```python
from unittest.mock import Mock, patch

@patch("fm_app_toolkit.data_loading.gcp.GCSReader")
def test_gcp_document_loading(mock_gcs_reader):
    # Mock the GCS reader
    mock_reader = Mock()
    mock_reader.load_data.return_value = [
        Document(text="Mocked content", metadata={"source": "test"})
    ]
    mock_gcs_reader.return_value = mock_reader
    
    # Test with mocked GCP repository
    repo = GCPDocumentRepository()
    documents = repo.load_documents(location="gs://test/test.txt")
    
    assert documents[0].text == "Mocked content"
```

### Integration Testing

```python
def test_end_to_end_with_local_data():
    """Test the entire pipeline with local test data."""
    # Prepare test documents
    repo = LocalDocumentRepository(input_dir="./tests/fixtures")
    
    # Run through your entire pipeline
    service = DocumentService(repo)
    results = service.process_documents()
    
    # Assert on results without needing external services
    assert results.success
```

## Best Practices

### 1. Configuration Management

Store repository configuration in environment variables or config files:

```python
# config.py
from pydantic_settings import BaseSettings

class DataLoadingConfig(BaseSettings):
    repository_type: str = "local"
    local_input_dir: str = "./data"
    gcs_bucket: str = ""
    gcs_prefix: str = ""
    
    class Config:
        env_prefix = "DATA_LOADING_"
```

### 2. Error Handling

Both repositories include proper error handling and logging:

```python
try:
    documents = repository.load_documents()
except Exception as e:
    logger.error(f"Failed to load documents: {e}")
    # Handle gracefully
```

### 3. Performance Considerations

- **LocalDocumentRepository**: Use `num_files_limit` for large directories
- **GCPDocumentRepository**: Use prefixes to limit the scope of file scanning
- Consider implementing caching for frequently accessed documents
- Use async/await patterns for I/O-bound operations when applicable

### 4. Testing First

Always start with `LocalDocumentRepository` during development:
1. Create sample test documents
2. Develop and test locally
3. Switch to cloud repository only when deploying

## Advanced Patterns

### Repository Factory

```python
from enum import Enum

class RepositoryType(Enum):
    LOCAL = "local"
    GCP = "gcp"

class RepositoryFactory:
    @staticmethod
    def create(repo_type: RepositoryType, **kwargs) -> DocumentRepository:
        if repo_type == RepositoryType.LOCAL:
            return LocalDocumentRepository(**kwargs)
        elif repo_type == RepositoryType.GCP:
            return GCPDocumentRepository(**kwargs)
        else:
            raise ValueError(f"Unknown repository type: {repo_type}")
```

### Composite Repository

```python
class CompositeRepository(DocumentRepository):
    """Load documents from multiple sources."""
    
    def __init__(self, repositories: list[DocumentRepository]):
        self.repositories = repositories
    
    def load_documents(self) -> list[Document]:
        all_documents = []
        for repo in self.repositories:
            all_documents.extend(repo.load_documents())
        return all_documents
```

## References

- 📚 [Cosmic Python - Chapter 2: Repository Pattern](https://www.cosmicpython.com/book/chapter_02_repository.html)
- 📚 [Cosmic Python - Chapter 6: Unit of Work Pattern](https://www.cosmicpython.com/book/chapter_06_uow.html) (related pattern)
- 📖 [LlamaIndex SimpleDirectoryReader](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/)
- 📖 [LlamaIndex GCS Reader](https://llamahub.ai/l/readers/llama-index-readers-gcs)
- 🏗️ [Martin Fowler - Repository Pattern](https://martinfowler.com/eaaCatalog/repository.html)

## Summary

The Repository pattern in this module provides:

1. **Abstraction** over document loading
2. **Testability** without external dependencies
3. **Flexibility** to swap implementations
4. **Separation** of business logic from infrastructure

By using `LocalDocumentRepository` for development and testing, and `GCPDocumentRepository` for production, we achieve a clean architecture that's both testable and scalable. This follows the key principle from [Cosmic Python](https://www.cosmicpython.com/book/chapter_02_repository.html):

> "We want our domain model to be free of infrastructure concerns."

The repository pattern ensures our document processing logic remains pure and testable, regardless of where the documents actually come from.