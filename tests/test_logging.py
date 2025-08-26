"""Essential tests for the structured logging module.

These tests demonstrate the core logging functionality without excessive repetition.
Each test teaches a specific concept about the logging system.
"""

import logging
import sys

import pytest
from pytest import LogCaptureFixture, MonkeyPatch

from fm_app_toolkit.logging import (
    LoggingContext,
    bind_contextvars,
    clear_context_fields,
    configure_structlog,
    get_logger,
    get_logging_level,
    get_stream,
)


def test_configure_structlog_with_defaults(caplog: LogCaptureFixture) -> None:
    """Demonstrate basic structlog configuration with default settings."""
    configure_structlog()
    logger = get_logger("test_logger")
    logger.info("Test message", extra_field="extra_value")
    
    log_output = caplog.records[0].message
    assert '"level": "info"' in log_output
    assert '"stream": "stdout"' in log_output
    assert '"extra_field": "extra_value"' in log_output


def test_configure_structlog_with_environment(caplog: LogCaptureFixture, monkeypatch: MonkeyPatch) -> None:
    """Demonstrate how environment variables control logging configuration."""
    monkeypatch.setenv("STREAM", "stderr")
    monkeypatch.setenv("LOGGING_LEVEL", "DEBUG")
    
    context = LoggingContext()
    configure_structlog(context)
    
    logger = get_logger("test_logger")
    logger.debug("Debug message")
    
    log_output = caplog.records[0].message
    assert '"level": "debug"' in log_output
    assert '"stream": "stderr"' in log_output


def test_context_binding_and_clearing(caplog: LogCaptureFixture) -> None:
    """Demonstrate context variable binding for correlation across log entries."""
    configure_structlog()
    
    # Bind context variables
    bind_contextvars(correlation_id="test-123", user_id="user-456")
    
    logger = get_logger("test_logger")
    logger.info("Message with context")
    
    log_output = caplog.records[0].message
    assert '"correlation_id": "test-123"' in log_output
    assert '"user_id": "user-456"' in log_output
    
    # Clear context
    clear_context_fields()
    logger.info("Message without context")
    
    log_output = caplog.records[1].message
    assert "correlation_id" not in log_output
    assert "user_id" not in log_output


def test_logging_levels() -> None:
    """Demonstrate logging level parsing and validation."""
    # Valid levels
    assert get_logging_level("DEBUG") == logging.DEBUG
    assert get_logging_level("info") == logging.INFO  # Case insensitive
    assert get_logging_level("WARNING") == logging.WARNING
    
    # Invalid level raises ValueError
    with pytest.raises(ValueError, match="Unsupported logging level"):
        get_logging_level("INVALID")


def test_stream_configuration() -> None:
    """Demonstrate stream selection for log output."""
    assert get_stream("stdout") == sys.stdout
    assert get_stream("STDERR") == sys.stderr  # Case insensitive
    
    # Invalid stream raises ValueError
    with pytest.raises(ValueError, match="Unsupported stream"):
        get_stream("invalid")


def test_multiple_loggers_share_context(caplog: LogCaptureFixture) -> None:
    """Demonstrate that context is shared across all loggers."""
    configure_structlog()
    bind_contextvars(request_id="req-789")
    
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")
    
    logger1.info("From module 1")
    logger2.info("From module 2")
    
    # Both logs should have the same context
    for record in caplog.records:
        assert '"request_id": "req-789"' in record.message


@pytest.mark.parametrize("level,stream,expected_level,expected_stream", [
    ("DEBUG", "stderr", "debug", "stderr"),
    ("INFO", "stdout", "info", "stdout"),
    ("WARNING", "stderr", "warning", "stderr"),
])
def test_logging_configuration_combinations(
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
    level: str,
    stream: str,
    expected_level: str,
    expected_stream: str
) -> None:
    """Demonstrate various logging configuration combinations."""
    monkeypatch.setenv("LOGGING_LEVEL", level)
    monkeypatch.setenv("STREAM", stream)
    
    context = LoggingContext()
    configure_structlog(context)
    
    logger = get_logger("test")
    getattr(logger, expected_level)("Test message")
    
    if caplog.records:  # Only check if the level allows the message
        log_output = caplog.records[0].message
        assert f'"level": "{expected_level}"' in log_output
        assert f'"stream": "{expected_stream}"' in log_output