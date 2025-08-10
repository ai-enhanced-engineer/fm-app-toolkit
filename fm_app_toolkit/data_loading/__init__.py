"""Data loading module for FM App Toolkit."""

from .base import DocumentRepository
from .gcp import GCPDocumentRepository
from .local import LocalDocumentRepository

__all__ = [
    "DocumentRepository",
    "LocalDocumentRepository",
    "GCPDocumentRepository",
]