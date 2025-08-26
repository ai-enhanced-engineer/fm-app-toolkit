"""Abstract base class for document repositories."""

from abc import ABC, abstractmethod

from llama_index.core import Document


class DocumentRepository(ABC):
    """Abstract interface for document loading from various sources."""

    @abstractmethod
    def load_documents(self, location: str) -> list[Document]:
        """Load documents from the specified location."""
        raise NotImplementedError
