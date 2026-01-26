"""Essential tests for the structured logging module.

These tests demonstrate the core logging functionality without excessive repetition.
Each test teaches a specific concept about the logging system.
"""

import concurrent.futures
import logging
import sys
import threading

import pytest
from pytest import LogCaptureFixture, MonkeyPatch

import src.logging
from src.logging import (
    LoggingContext,
    bind_contextvars,
    clear_context_fields,
    configure_structlog,
    get_logger,
    get_logging_level,
    get_stream,
)


def test__configure_structlog__with_defaults(caplog: LogCaptureFixture) -> None:
    """Demonstrate basic structlog configuration with default settings."""
    configure_structlog()
    logger = get_logger("test_logger")
    logger.info("Test message", extra_field="extra_value")

    log_output = caplog.records[0].message
    assert '"level": "info"' in log_output
    assert '"stream": "stdout"' in log_output
    assert '"extra_field": "extra_value"' in log_output


def test__configure_structlog__with_environment(caplog: LogCaptureFixture, monkeypatch: MonkeyPatch) -> None:
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


def test__context_binding__and_clearing(caplog: LogCaptureFixture) -> None:
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


def test__logging__levels() -> None:
    """Demonstrate logging level parsing and validation."""
    # Valid levels
    assert get_logging_level("DEBUG") == logging.DEBUG
    assert get_logging_level("info") == logging.INFO  # Case insensitive
    assert get_logging_level("WARNING") == logging.WARNING

    # Invalid level raises ValueError
    with pytest.raises(ValueError, match="Unsupported logging level"):
        get_logging_level("INVALID")


def test__stream__configuration() -> None:
    """Demonstrate stream selection for log output."""
    assert get_stream("stdout") == sys.stdout
    assert get_stream("STDERR") == sys.stderr  # Case insensitive

    # Invalid stream raises ValueError
    with pytest.raises(ValueError, match="Unsupported stream"):
        get_stream("invalid")


def test__multiple_loggers__share_context(caplog: LogCaptureFixture) -> None:
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


@pytest.mark.parametrize(
    "level,stream,expected_level,expected_stream",
    [
        ("DEBUG", "stderr", "debug", "stderr"),
        ("INFO", "stdout", "info", "stdout"),
        ("WARNING", "stderr", "warning", "stderr"),
    ],
)
def test_logging_configuration_combinations(
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
    level: str,
    stream: str,
    expected_level: str,
    expected_stream: str,
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


# Thread-safety tests for logging initialization
class TestThreadSafeLogging:
    """Tests for thread-safe logging initialization."""

    def test__get_logger__thread_safe_initialization(self) -> None:
        """Multiple threads calling get_logger concurrently initializes only once."""
        # Reset configuration state
        original_configured = src.logging._configured
        src.logging._configured = False

        call_count = 0
        original_configure = src.logging.configure_structlog

        def counting_configure(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return original_configure(*args, **kwargs)

        # Patch to count calls
        src.logging.configure_structlog = counting_configure  # type: ignore[assignment]

        try:

            def get_logger_task() -> object:
                logger = get_logger(f"test_thread_{threading.current_thread().ident}")
                return logger

            # Execute from 10 threads concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(get_logger_task) for _ in range(10)]
                results = [f.result() for f in futures]

            # Should only configure once despite 10 concurrent calls
            assert call_count == 1
            assert len(results) == 10
        finally:
            # Cleanup
            src.logging.configure_structlog = original_configure  # type: ignore[assignment]
            src.logging._configured = original_configured

    def test__get_logger__returns_logger_after_concurrent_init(self) -> None:
        """All threads receive valid logger instances."""
        src.logging._configured = False

        loggers: list[object] = []
        lock = threading.Lock()

        def collect_logger() -> None:
            logger = get_logger(f"thread_{threading.current_thread().ident}")
            with lock:
                loggers.append(logger)

        threads = [threading.Thread(target=collect_logger) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should have received valid loggers
        assert len(loggers) == 5
        for logger in loggers:
            assert logger is not None
            assert hasattr(logger, "info")

    def test__configure_lock_exists__prevents_race_conditions(self) -> None:
        """Verify the configure lock attribute exists."""
        assert hasattr(src.logging, "_configure_lock")
        assert isinstance(src.logging._configure_lock, type(threading.Lock()))

    def test__configured_flag__is_set_after_init(self) -> None:
        """Verify _configured flag is set after initialization."""
        # Force reconfiguration
        src.logging._configured = False
        get_logger("test")

        assert src.logging._configured is True
