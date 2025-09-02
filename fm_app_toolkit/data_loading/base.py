"""Abstract base class for document repositories."""

from abc import ABC, abstractmethod

import pandas as pd
from llama_index.core import Document
from pydantic import BaseModel


class BaseRepository(ABC):
    """Abstract interface for data loading from various sources."""

    @abstractmethod
    def load_data(self, path: str) -> pd.DataFrame:
        raise NotImplementedError


class DocumentRepository(BaseModel):
    """Abstract interface for document loading from various sources."""

    @abstractmethod
    def load_documents(self, location: str) -> list[Document]:
        raise NotImplementedError
