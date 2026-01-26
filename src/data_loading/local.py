"""Local filesystem document repository implementation."""

import pandas as pd
from llama_index.core import Document, SimpleDirectoryReader
from pydantic import BaseModel, validate_call

from .base import BaseRepository, DocumentRepository


class LocalRepository(BaseRepository):
    """Load CSV data from local filesystem."""

    def load_data(self, path: str) -> pd.DataFrame:
        """Load CSV data from file."""
        return pd.read_csv(path)


class LocalDocumentRepository(DocumentRepository, BaseModel):
    """Load documents from local filesystem using SimpleDirectoryReader."""

    input_dir: str
    recursive: bool = True
    required_exts: list[str] | None = None
    exclude_hidden: bool = True
    num_files_limit: int | None = None

    @validate_call
    def load_documents(self, location: str) -> list[Document]:
        """Load documents from filesystem directory."""
        reader = SimpleDirectoryReader(
            input_dir=location,
            recursive=self.recursive,
            required_exts=self.required_exts,
            exclude_hidden=self.exclude_hidden,
            num_files_limit=self.num_files_limit,
        )
        return reader.load_data()
