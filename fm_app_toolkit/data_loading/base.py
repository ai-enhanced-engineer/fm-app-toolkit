"""Abstract base class for document repositories."""

from abc import ABC, abstractmethod

from llama_index.core import Document


class DocumentRepository(ABC):
    """Abstract interface for document loading from various sources."""

    @abstractmethod
    def load_documents(self) -> list[Document]:
        """Load all documents from the repository."""
        raise NotImplementedError
