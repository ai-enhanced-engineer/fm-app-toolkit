"""Local filesystem document repository implementation."""

from typing import Optional

from llama_index.core import Document, SimpleDirectoryReader

from fm_app_toolkit.logging import get_logger

from .base import DocumentRepository

logger = get_logger(__name__)


class LocalDocumentRepository(DocumentRepository):
    """Load documents from local filesystem using LlamaIndex SimpleDirectoryReader."""
    
    def __init__(
        self,
        input_dir: str,
        recursive: bool = True,
        required_exts: Optional[list[str]] = None,
        exclude_hidden: bool = True,
        num_files_limit: Optional[int] = None,
    ):
        self.input_dir = input_dir
        self.recursive = recursive
        self.required_exts = required_exts
        self.exclude_hidden = exclude_hidden
        self.num_files_limit = num_files_limit
        
        logger.info(
            "Initializing LocalDocumentRepository",
            input_dir=input_dir,
            recursive=recursive,
            required_exts=required_exts,
        )
        
    def load_documents(self) -> list[Document]:
        """Load documents using SimpleDirectoryReader."""
        try:
            reader = SimpleDirectoryReader(
                input_dir=self.input_dir,
                recursive=self.recursive,
                required_exts=self.required_exts,
                exclude_hidden=self.exclude_hidden,
                num_files_limit=self.num_files_limit,
            )
            documents = reader.load_data()
            logger.info(f"Successfully loaded {len(documents)} documents from {self.input_dir}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load documents from {self.input_dir}: {e}")
            raise