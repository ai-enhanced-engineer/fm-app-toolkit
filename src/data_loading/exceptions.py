"""Custom exceptions for data loading operations.

This module provides a hierarchy of exceptions for GCS and other data loading
operations, enabling structured error handling and better error messages.
"""


class DataLoadingError(Exception):
    """Base exception for all data loading errors."""

    pass


class GCSError(DataLoadingError):
    """Base exception for GCS-related errors."""

    pass


class GCSURIError(GCSError):
    """Raised when a GCS URI is invalid or malformed.

    Example:
        >>> raise GCSURIError("path/to/file", "URI must start with gs://")
        Traceback (most recent call last):
        ...
        GCSURIError: Invalid GCS URI 'path/to/file': URI must start with gs://
    """

    def __init__(self, uri: str, reason: str) -> None:
        self.uri = uri
        self.reason = reason
        super().__init__(f"Invalid GCS URI '{uri}': {reason}")


class GCSLoadError(GCSError):
    """Raised when loading documents from GCS fails.

    Example:
        >>> raise GCSLoadError("gs://bucket/path", "Permission denied")
        Traceback (most recent call last):
        ...
        GCSLoadError: Failed to load from 'gs://bucket/path': Permission denied
    """

    def __init__(self, location: str, reason: str) -> None:
        self.location = location
        self.reason = reason
        super().__init__(f"Failed to load from '{location}': {reason}")
