"""Google Cloud Storage document repository implementation."""

from typing import Optional

from llama_index.core import Document
from llama_index.readers.gcs import GCSReader

from fm_app_toolkit.logging import get_logger

from .base import DocumentRepository

logger = get_logger(__name__)


class GCPDocumentRepository(DocumentRepository):
    """Load documents from Google Cloud Storage using LlamaIndex GCSReader."""
    
    def __init__(
        self,
        bucket: str,
        key: Optional[str] = None,
        prefix: Optional[str] = None,
        service_account_key: Optional[dict] = None,
    ):
        if not key and not prefix:
            raise ValueError("Either 'key' or 'prefix' must be provided")
        
        self.bucket = bucket
        self.key = key
        self.prefix = prefix
        self.service_account_key = service_account_key
        
        logger.info(
            "Initializing GCPDocumentRepository",
            bucket=bucket,
            key=key,
            prefix=prefix,
        )
        
    def load_documents(self) -> list[Document]:
        """Load documents from GCS using GCSReader."""
        try:
            reader_kwargs = {"bucket": self.bucket}
            
            if self.key:
                reader_kwargs["key"] = self.key
            elif self.prefix:
                reader_kwargs["prefix"] = self.prefix
                
            if self.service_account_key:
                reader_kwargs["service_account_key"] = self.service_account_key
                
            reader = GCSReader(**reader_kwargs)
            documents = reader.load_data()
            
            logger.info(
                f"Successfully loaded {len(documents)} documents from GCS",
                bucket=self.bucket,
                key=self.key,
                prefix=self.prefix,
            )
            return documents
        except Exception as e:
            logger.error(f"Failed to load documents from GCS bucket {self.bucket}: {e}")
            raise