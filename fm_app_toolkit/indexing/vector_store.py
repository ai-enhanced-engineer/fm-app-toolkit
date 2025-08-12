"""VectorStore index implementation."""

from typing import Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import validate_call

from fm_app_toolkit.logging import get_logger

from .base import BaseIndexer

logger = get_logger(__name__)


class VectorStoreIndexer(BaseIndexer):
    """Create vector store indexes from documents using LlamaIndex.
    
    Note: insert_batch_size affects memory usage during indexing. The default
    of 2048 works well for most document sets. Larger batches use more memory
    but may be faster for large corpuses.
    
    Empty document lists create valid but empty indexes that can still be queried.
    """

    def __init__(
        self,
        show_progress: bool = False,
        insert_batch_size: int = 2048,
    ):
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
        embed_model: Optional[BaseEmbedding] = None,
    ) -> VectorStoreIndex:
        """Create a vector store index from documents.
        
        Pydantic automatically validates that documents is a list.
        """
        try:
            logger.info(f"Creating index from {len(documents)} documents")
            
            # Create index with optional embedding model
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embed_model,
                show_progress=self.show_progress,
                insert_batch_size=self.insert_batch_size,
            )
            
            logger.info(f"Successfully created index with {len(documents)} documents")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise