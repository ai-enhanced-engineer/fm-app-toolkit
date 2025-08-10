"""Structured logging configuration for the AI Test Lab project.

This module provides structured logging using structlog, with support for
JSON and key-value formats, context variables, and proper log field handling.
"""

import logging
import sys
from typing import Any, Optional, TextIO

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog.types import EventDict, WrappedLogger

LOG_SPECIFIC_FIELDS = {
    "timestamp",
    "thread",
    "logger",
    "message",
    "context",
    "trace_id",
    "trace_flags",
    "span_id",
}
CONTEXT_FIELDS = {
    "level",
    "stream",
}


class LoggingContext(BaseSettings):
    stream: str = Field(
        default="stdout",
        description="The log stream used (stdout or stderr)",
        json_schema_extra={"env_names": ["STREAM"]},
    )
    logging_level: str = Field(
        default="INFO",
        description="The logging level of the application",
        json_schema_extra={"env_names": ["LOGGING_LEVEL"]},
    )
    log_format: str = Field(
        default="json",
        description="The log output format (json or keyvalue)",
        json_schema_extra={"env_names": ["LOG_FORMAT"]},
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )


def get_logging_level(level: str) -> int:
    """Return the logging level based on the provided string."""
    level = level.upper()
    if level == "DEBUG":
        return logging.DEBUG
    elif level == "INFO":
        return logging.INFO
    elif level == "WARNING":
        return logging.WARNING
    elif level == "ERROR":
        return logging.ERROR
    elif level == "CRITICAL":
        return logging.CRITICAL
    else:
        raise ValueError(f"Unsupported logging level: {level}")


def get_stream(stream: str) -> TextIO:
    """Return the appropriate stream based on the provided string."""
    if stream.lower() == "stdout":
        return sys.stdout
    elif stream.lower() == "stderr":
        return sys.stderr
    else:
        raise ValueError(f"Unsupported stream: {stream}")


def _process_log_fields(logger: WrappedLogger, log_method: str, event_dict: EventDict) -> EventDict:
    # Rename "event" to "message" or set it to an empty string if not present
    event_dict["message"] = event_dict.pop("event", "")
    event_dict["context"] = structlog.contextvars.get_contextvars().get("context", "default")

    allowed_keys = LOG_SPECIFIC_FIELDS | CONTEXT_FIELDS

    # Create the "extra" dictionary for unexpected keys and pop them from event_dict
    extra_fields = {key: event_dict.pop(key) for key in list(event_dict.keys()) if key not in allowed_keys}

    # Add the "extra" dictionary to event_dict if it contains any fields
    if extra_fields:
        event_dict["extra"] = extra_fields

    # Add trace-related fields if present in event_dict
    if "thread" in event_dict:
        event_dict["thread"] = event_dict.pop("thread")
    if "trace_id" in event_dict:
        event_dict["trace_id"] = event_dict.pop("trace_id")
    if "trace_flags" in event_dict:
        event_dict["trace_flags"] = event_dict.pop("trace_flags")
    if "span_id" in event_dict:
        event_dict["span_id"] = event_dict.pop("span_id")

    return event_dict


def configure_structlog(context: Optional[LoggingContext] = None) -> None:
    if context is None:
        context = LoggingContext()

    logging.basicConfig(
        format="%(message)s",
        level=get_logging_level(context.logging_level),
        stream=get_stream(context.stream),
    )

    structlog_logger = logging.getLogger()
    structlog_logger.setLevel(get_logging_level(context.logging_level))

    renderer = (
        structlog.processors.JSONRenderer()
        if context.log_format.lower() == "json"
        else structlog.processors.KeyValueRenderer(key_order=["timestamp", "level", "logger", "message"])
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.contextvars.merge_contextvars,
            _process_log_fields,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(get_logging_level(context.logging_level)),
        cache_logger_on_first_use=True,
    )

    try:
        set_context_fields(context)
    except Exception as e:
        logging.error(f"Failed to configure structlog context: {e}")
        raise


def set_context_fields(context: LoggingContext) -> None:
    structlog.contextvars.bind_contextvars(
        stream=context.stream,
    )


def clear_context_fields() -> None:
    structlog.contextvars.clear_contextvars()


def bind_contextvars(**kwargs: Any) -> None:
    """Bind context variables to the current context.

    This is a wrapper around structlog.contextvars.bind_contextvars
    to keep all structlog interaction in this module.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def get_contextvars() -> dict[str, Any]:
    """Get all context variables from the current context.

    Returns:
        Dictionary of all context variables in the current context.
    """
    return structlog.contextvars.get_contextvars()


def get_correlation_id() -> str:
    """Get correlation ID from context or return default.

    Returns:
        The correlation_id from context, or "unknown" if not found.
    """
    contextvars = structlog.contextvars.get_contextvars()
    return str(contextvars.get("correlation_id", "unknown"))


# Configure the logger when the module is imported
_configured = False


def get_logger(name: str = "") -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: The name of the logger. If empty, uses the module name.

    Returns:
        A configured structlog BoundLogger instance.
    """
    global _configured
    if not _configured:
        configure_structlog()
        _configured = True

    if not name:
        name = __name__
    return structlog.get_logger(name)  # type: ignore
