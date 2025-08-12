"""Google Cloud Storage document repository implementation."""

from typing import Any, Optional

from llama_index.core import Document
from llama_index.readers.gcs import GCSReader
from pydantic import validate_call

from fm_app_toolkit.logging import get_logger

from .base import DocumentRepository

logger = get_logger(__name__)


def _parse_gcs_uri(uri: str) -> dict[str, str]:
    """Parse GCS URI into bucket and path components.
    
    Examples:
        gs://bucket -> {"bucket": "bucket"}
        gs://bucket/file.txt -> {"bucket": "bucket", "key": "file.txt"}
        gs://bucket/dir/ -> {"bucket": "bucket", "prefix": "dir/"}
    """
    if not uri.startswith("gs://"):
        raise ValueError("GCS location must start with gs://")
    
    # Remove gs:// prefix and split into bucket and path
    path_without_prefix = uri[5:]
    parts = path_without_prefix.split("/", 1)
    
    result = {"bucket": parts[0]}
    
    # If there's a path after the bucket
    if len(parts) > 1 and parts[1]:
        object_path = parts[1]
        if object_path.endswith("/"):
            result["prefix"] = object_path
        else:
            result["key"] = object_path
    
    return result


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
            # Parse the GCS URI
            reader_kwargs = _parse_gcs_uri(location)
            
            # Add service account key if provided
            if self.service_account_key:
                reader_kwargs["service_account_key"] = self.service_account_key
                
            reader = GCSReader(**reader_kwargs)
            documents: list[Document] = reader.load_data()
            
            logger.info(f"Successfully loaded {len(documents)} documents from {location}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load documents from {location}: {e}")
            raise
