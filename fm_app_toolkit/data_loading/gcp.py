"""Google Cloud Storage document repository implementation."""

from typing import Any, Optional

from llama_index.core import Document
from llama_index.readers.gcs import GCSReader
from pydantic import validate_call

from fm_app_toolkit.logging import get_logger

from .base import DocumentRepository

logger = get_logger(__name__)


class GCPDocumentRepository(DocumentRepository):
    """Load documents from Google Cloud Storage using LlamaIndex GCSReader."""

    def __init__(
        self,
        service_account_key: Optional[dict[str, Any]] = None,
    ):
        self.service_account_key = service_account_key
        logger.info("Initializing GCPDocumentRepository")

    @validate_call
    def load_documents(self, location: str) -> list[Document]:
        """Load documents from GCS path.
        
        Format: gs://bucket/path or gs://bucket/prefix/
        """
        try:
            if not location.startswith("gs://"):
                raise ValueError(f"GCS location must start with gs://")
            
            # Simple parsing: gs://bucket/rest_of_path
            path = location[5:]  # Remove gs://
            parts = path.split("/", 1)
            bucket = parts[0]
            
            reader_kwargs: dict[str, Any] = {"bucket": bucket}
            
            # If there's a path after bucket
            if len(parts) > 1:
                path = parts[1]
                if path.endswith("/"):
                    reader_kwargs["prefix"] = path
                else:
                    reader_kwargs["key"] = path
            
            if self.service_account_key:
                reader_kwargs["service_account_key"] = self.service_account_key
                
            reader = GCSReader(**reader_kwargs)
            documents: list[Document] = reader.load_data()
            
            logger.info(f"Successfully loaded {len(documents)} documents from {location}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load documents from {location}: {e}")
            raise
