"""VectorStore index implementation."""

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import validate_call

from src.logging import get_logger

from .base import DocumentIndexer

logger = get_logger(__name__)


class VectorStoreIndexer(DocumentIndexer):
    """Index documents for similarity search using embeddings."""

    def __init__(
        self,
        show_progress: bool = False,
        insert_batch_size: int = 2048,
    ) -> None:
        self.show_progress = show_progress
        self.insert_batch_size = insert_batch_size

        logger.info(
            "Initializing VectorStoreIndexer",
            show_progress=show_progress,
            insert_batch_size=insert_batch_size,
        )

    @validate_call
    def create_index(
        self,
        documents: list[Document],
        embed_model: BaseEmbedding | None = None,
    ) -> VectorStoreIndex:
        """Build searchable index from documents."""
        logger.info("Creating index from documents", document_count=len(documents))

        try:
            # Create index with optional embedding model
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embed_model,
                show_progress=self.show_progress,
                insert_batch_size=self.insert_batch_size,
            )

            logger.info("Successfully created index", document_count=len(documents))
            return index

        except Exception as e:
            logger.error("Failed to create index", document_count=len(documents), error=str(e))
            raise
