"""Local filesystem document repository implementation."""

from typing import Optional

import pandas as pd
from llama_index.core import Document, SimpleDirectoryReader
from pydantic import validate_call

from .base import BaseRepository, DocumentRepository


class LocalRepository(BaseRepository):
    """Load CSV data from local filesystem."""

    def load_data(self, path: str) -> pd.DataFrame:
        """Load CSV data from the specified file path."""
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            raise

class LocalDocumentRepository(DocumentRepository):
    """Load documents from local filesystem using LlamaIndex SimpleDirectoryReader."""
    
    input_dir: str
    recursive: bool = True
    required_exts: Optional[list[str]] = None
    exclude_hidden: bool = True
    num_files_limit: Optional[int] = None

    @validate_call
    def load_documents(self, location: str) -> list[Document]:
        """Load documents from local filesystem path."""
        try:
            reader = SimpleDirectoryReader(
                input_dir=location,
                recursive=self.recursive,
                required_exts=self.required_exts,
                exclude_hidden=self.exclude_hidden,
                num_files_limit=self.num_files_limit,
            )
            documents = reader.load_data()
            return documents
        except Exception:
            raise
