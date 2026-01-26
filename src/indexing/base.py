"""Abstract base class for document indexers."""

from abc import ABC, abstractmethod

from llama_index.core import Document, PropertyGraphIndex, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding


class DocumentIndexer(ABC):
    """Abstract interface for creating indexes from documents."""

    @abstractmethod
    def create_index(
        self,
        documents: list[Document],
        embed_model: BaseEmbedding | None = None,
    ) -> VectorStoreIndex | PropertyGraphIndex: ...
