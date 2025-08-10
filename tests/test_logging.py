"""Tests for the structured logging module."""

import logging
import sys

import pytest
from pytest import LogCaptureFixture, MonkeyPatch
from structlog.stdlib import BoundLogger

from fm_app_toolkit.logging import (
    LoggingContext,
    bind_contextvars,
    clear_context_fields,
    configure_structlog,
    get_contextvars,
    get_correlation_id,
    get_logger,
    get_logging_level,
    get_stream,
    set_context_fields,
)


def test__configure_structlog__default_context_binding(caplog: LogCaptureFixture) -> None:
    """Test that default context is properly bound when configuring structlog."""
    configure_structlog()
    logger = get_logger("test_logger")
    logger.info("Test message")

    log_output: str = caplog.records[0].message
    assert '"stream": "stdout"' in log_output
    assert '"level": "info"' in log_output


def test__configure_structlog__custom_context_binding(caplog: LogCaptureFixture, monkeypatch: MonkeyPatch) -> None:
    """Test that custom context from environment variables is properly bound."""
    monkeypatch.setenv("STREAM", "stderr")
    monkeypatch.setenv("LOGGING_LEVEL", "DEBUG")

    custom_context = LoggingContext()
    configure_structlog(context=custom_context)
    logger = get_logger("test_logger")
    logger.debug("Test message")

    log_output: str = caplog.records[0].message
    assert '"stream": "stderr"' in log_output
    assert '"level": "debug"' in log_output


def test__configure_structlog__keyvalue_format(caplog: LogCaptureFixture, monkeypatch: MonkeyPatch) -> None:
    """Test that key-value format is properly applied when configured."""
    monkeypatch.setenv("LOG_FORMAT", "keyvalue")
    configure_structlog()
    logger = get_logger("kv_logger")
    logger.info("Format test")

    log_output: str = caplog.records[0].message
    assert "message='Format test'" in log_output  # KeyValueRenderer uses single quotes
    assert '"message":' not in log_output  # Should not have JSON format


def test__log_format__includes_all_expected_fields(caplog: LogCaptureFixture) -> None:
    """Test that all expected fields are included in the log output."""
    configure_structlog()
    logger = get_logger("test_logger")

    logger.info(
        "Formatted log test",
        thread="test-thread",
        trace_id="test-trace-id",
        trace_flags="test-flags",
        span_id="test-span-id",
    )
    log_output_with_all_fields: str = caplog.records[0].message

    assert '"message": "Formatted log test"' in log_output_with_all_fields
    assert '"thread": "test-thread"' in log_output_with_all_fields
    assert '"trace_id": "test-trace-id"' in log_output_with_all_fields
    assert '"trace_flags": "test-flags"' in log_output_with_all_fields
    assert '"span_id": "test-span-id"' in log_output_with_all_fields


def test__clear_context_fields__removes_all_context(caplog: LogCaptureFixture) -> None:
    """Test that clearing context fields removes all context variables."""
    configure_structlog()
    set_context_fields(LoggingContext(stream="stderr", logging_level="DEBUG"))
    clear_context_fields()

    logger = get_logger("test_logger")
    logger.info("Test after clearing")

    log_output: str = caplog.records[0].message
    # After clearing, only the default context should remain
    assert '"context": "default"' in log_output


def test__get_logger__uses_correct_logger_name(caplog: LogCaptureFixture) -> None:
    """Test that get_logger uses the correct logger name."""
    configure_structlog()
    logger = get_logger("custom_logger")
    logger.info("Test logger name")

    log_output: str = caplog.records[0].message
    assert '"logger": "custom_logger"' in log_output


def test__get_logger__uses_module_name_when_empty(caplog: LogCaptureFixture) -> None:
    """Test that get_logger uses module name when no name is provided."""
    configure_structlog()
    logger = get_logger("")  # Empty string should use module name
    logger.info("Test default name")

    log_output: str = caplog.records[0].message
    assert '"logger": "fm_app_toolkit.logging"' in log_output


def test__process_log_fields__conditionally_adds_thread_and_trace_fields(caplog: LogCaptureFixture) -> None:
    """Test that thread and trace fields are conditionally added to logs."""
    configure_structlog()
    logger: BoundLogger = get_logger("test_logger")

    logger.info(
        "Health check with thread and trace",
        thread="test-thread",
        trace_id="test-trace-id",
        trace_flags="test-flags",
        span_id="test-span-id",
    )
    log_output_with_thread_and_trace: str = caplog.records[0].message

    assert '"thread": "test-thread"' in log_output_with_thread_and_trace
    assert '"trace_id": "test-trace-id"' in log_output_with_thread_and_trace
    assert '"trace_flags": "test-flags"' in log_output_with_thread_and_trace
    assert '"span_id": "test-span-id"' in log_output_with_thread_and_trace


def test__process_log_fields__handles_extra_fields(caplog: LogCaptureFixture) -> None:
    """Test that extra fields are properly grouped in the 'extra' dictionary."""
    configure_structlog()
    logger: BoundLogger = get_logger("test_logger")

    logger.info(
        "Log with extra fields",
        custom_field_1="extra_value_1",
        custom_field_2="extra_value_2",
        user_id=123,
        request_id="abc-123",
    )

    log_output: str = caplog.records[0].message

    assert '"extra": {' in log_output
    assert '"custom_field_1": "extra_value_1"' in log_output
    assert '"custom_field_2": "extra_value_2"' in log_output
    assert '"user_id": 123' in log_output
    assert '"request_id": "abc-123"' in log_output
    assert '"message": "Log with extra fields"' in log_output
    assert '"logger": "test_logger"' in log_output


def test__logging_context__extracts_environment_variables(monkeypatch: MonkeyPatch) -> None:
    """Test that LoggingContext properly extracts environment variables."""
    monkeypatch.setenv("STREAM", "stderr")
    monkeypatch.setenv("LOGGING_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "keyvalue")

    context = LoggingContext()

    assert context.stream == "stderr"
    assert context.logging_level == "DEBUG"
    assert context.log_format == "keyvalue"


def test__logging_context__uses_defaults(monkeypatch: MonkeyPatch) -> None:
    """Test that LoggingContext uses default values when env vars are not set."""
    # Clear the environment variables to prevent overrides
    monkeypatch.delenv("LOGGING_LEVEL", raising=False)
    monkeypatch.delenv("STREAM", raising=False)
    monkeypatch.delenv("LOG_FORMAT", raising=False)

    context = LoggingContext()

    assert context.stream == "stdout"
    assert context.logging_level == "INFO"
    assert context.log_format == "json"


def test__get_logging_level__valid_levels() -> None:
    """Test that get_logging_level returns correct logging levels."""
    assert get_logging_level("DEBUG") == logging.DEBUG
    assert get_logging_level("INFO") == logging.INFO
    assert get_logging_level("WARNING") == logging.WARNING
    assert get_logging_level("ERROR") == logging.ERROR
    assert get_logging_level("CRITICAL") == logging.CRITICAL


def test__get_logging_level__case_insensitive() -> None:
    """Test that get_logging_level is case insensitive."""
    assert get_logging_level("debug") == logging.DEBUG
    assert get_logging_level("Info") == logging.INFO
    assert get_logging_level("warning") == logging.WARNING


def test__get_logging_level__invalid_level() -> None:
    """Test that get_logging_level raises ValueError for invalid levels."""
    with pytest.raises(ValueError, match="Unsupported logging level: INVALID"):
        get_logging_level("INVALID")


def test__get_stream__valid_streams() -> None:
    """Test that get_stream returns correct stream objects."""
    assert get_stream("stdout") is sys.stdout
    assert get_stream("stderr") is sys.stderr


def test__get_stream__case_insensitive() -> None:
    """Test that get_stream is case insensitive."""
    assert get_stream("STDOUT") is sys.stdout
    assert get_stream("StdErr") is sys.stderr


def test__get_stream__invalid_stream() -> None:
    """Test that get_stream raises ValueError for invalid streams."""
    with pytest.raises(ValueError, match="Unsupported stream: invalid-stream"):
        get_stream("invalid-stream")


def test__configure_structlog__logs_correct_level_and_stream(monkeypatch: MonkeyPatch) -> None:
    """Test that structlog is configured with correct level and stream."""
    monkeypatch.setenv("LOGGING_LEVEL", "DEBUG")
    monkeypatch.setenv("STREAM", "stdout")

    custom_context = LoggingContext()
    configure_structlog(context=custom_context)

    python_logger = logging.getLogger("test_logger")
    assert python_logger.getEffectiveLevel() == logging.DEBUG


def test__bind_contextvars__adds_context_variables(caplog: LogCaptureFixture) -> None:
    """Test that bind_contextvars adds context variables to logs."""
    configure_structlog()
    
    # Bind some context variables
    bind_contextvars(
        correlation_id="test-correlation-123",
        user_id="user-456",
        request_path="/api/test"
    )
    
    logger = get_logger("test_logger")
    logger.info("Test with context vars")
    
    log_output: str = caplog.records[0].message
    
    # These should be in the extra fields
    assert "test-correlation-123" in log_output
    assert "user-456" in log_output
    assert "/api/test" in log_output
    
    # Clear for other tests
    clear_context_fields()


def test__get_contextvars__returns_all_context_variables() -> None:
    """Test that get_contextvars returns all bound context variables."""
    configure_structlog()
    clear_context_fields()
    
    # Start with empty context
    assert get_contextvars() == {}
    
    # Bind some variables
    bind_contextvars(
        correlation_id="test-123",
        user_id="user-789",
        session_id="session-abc"
    )
    
    contextvars = get_contextvars()
    assert contextvars["correlation_id"] == "test-123"
    assert contextvars["user_id"] == "user-789"
    assert contextvars["session_id"] == "session-abc"
    
    # Clear for other tests
    clear_context_fields()


def test__get_correlation_id__returns_correct_value() -> None:
    """Test that get_correlation_id returns the correct correlation ID."""
    configure_structlog()
    clear_context_fields()
    
    # Should return "unknown" when not set
    assert get_correlation_id() == "unknown"
    
    # Bind a correlation ID
    bind_contextvars(correlation_id="test-correlation-999")
    assert get_correlation_id() == "test-correlation-999"
    
    # Clear for other tests
    clear_context_fields()


def test__multiple_loggers__share_context(caplog: LogCaptureFixture) -> None:
    """Test that multiple logger instances share the same context variables."""
    configure_structlog()
    clear_context_fields()
    
    # Bind context that should be shared
    bind_contextvars(shared_context="shared-value-123")
    
    logger1 = get_logger("logger1")
    logger2 = get_logger("logger2")
    
    logger1.info("Message from logger1")
    logger2.info("Message from logger2")
    
    # Both log messages should contain the shared context
    assert all("shared-value-123" in record.message for record in caplog.records)
    
    # Clear for other tests
    clear_context_fields()


def test__logger_with_different_levels(caplog: LogCaptureFixture, monkeypatch: MonkeyPatch) -> None:
    """Test that logger respects different logging levels."""
    # Set to WARNING level
    monkeypatch.setenv("LOGGING_LEVEL", "WARNING")
    configure_structlog()
    
    logger = get_logger("test_logger")
    
    # These should not appear in logs
    logger.debug("Debug message")
    logger.info("Info message")
    
    # These should appear
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    # Only WARNING and above should be captured
    assert len(caplog.records) == 3
    messages = [record.message for record in caplog.records]
    assert any("Warning message" in msg for msg in messages)
    assert any("Error message" in msg for msg in messages)
    assert any("Critical message" in msg for msg in messages)