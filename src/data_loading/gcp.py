"""Google Cloud Storage document repository implementation."""

from typing import Any

from llama_index.core import Document
from llama_index.readers.gcs import GCSReader
from pydantic import BaseModel, validate_call

from src.logging import get_logger

from .base import DocumentRepository
from .exceptions import GCSLoadError, GCSURIError

logger = get_logger(__name__)


def _parse_gcs_uri(uri: str) -> dict[str, Any]:
    """Parse GCS URI into bucket and path components for GCSReader.

    Raises:
        GCSURIError: If the URI doesn't start with gs:// or is otherwise invalid.
    """
    if not uri.startswith("gs://"):
        raise GCSURIError(uri, "URI must start with gs://")

    # Remove gs:// prefix and split into bucket and path
    path_without_prefix = uri[5:]
    parts = path_without_prefix.split("/", 1)

    result: dict[str, Any] = {"bucket": parts[0]}

    # If there's a path after the bucket
    if len(parts) > 1 and parts[1]:
        object_path = parts[1]
        if object_path.endswith("/"):
            result["prefix"] = object_path
        else:
            result["key"] = object_path

    return result


class GCPDocumentRepository(DocumentRepository, BaseModel):
    """Load documents from Google Cloud Storage using GCSReader."""

    service_account_key: dict[str, Any] | None = None

    @validate_call
    def load_documents(self, location: str) -> list[Document]:
        """Load documents from GCS bucket using gs:// URI format.

        Raises:
            GCSURIError: If the location URI is invalid.
            GCSLoadError: If loading documents fails.
        """
        try:
            # Parse the GCS URI
            reader_kwargs = _parse_gcs_uri(location)
            logger.debug("Parsed GCS URI", location=location, bucket=reader_kwargs.get("bucket"))

            # Add service account key if provided
            if self.service_account_key:
                reader_kwargs["service_account_key"] = self.service_account_key

            reader = GCSReader(**reader_kwargs)
            documents: list[Document] = reader.load_data()
            logger.info("Successfully loaded documents from GCS", location=location, count=len(documents))

            return documents
        except GCSURIError:
            logger.error("Invalid GCS URI format", location=location)
            raise
        except (OSError, IOError, PermissionError) as e:
            # File system and permission errors
            logger.error("Failed to load documents from GCS", location=location, error=str(e))
            raise GCSLoadError(location, str(e)) from e
        except (ConnectionError, TimeoutError) as e:
            # Network-related errors
            logger.error("Network error loading documents from GCS", location=location, error=str(e))
            raise GCSLoadError(location, f"Network error: {e}") from e
        except (ValueError, TypeError, KeyError) as e:
            # Data parsing and configuration errors
            logger.error("Configuration error loading documents from GCS", location=location, error=str(e))
            raise GCSLoadError(location, f"Configuration error: {e}") from e
        except RuntimeError as e:
            # Runtime errors from GCS client
            logger.error("Runtime error loading documents from GCS", location=location, error=str(e))
            raise GCSLoadError(location, str(e)) from e
