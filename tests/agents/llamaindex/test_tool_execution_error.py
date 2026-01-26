"""Tests for MinimalReActAgent error handling - ToolExecutionError."""

import pytest

from src.agents.llamaindex.minimal_react import ToolExecutionError


class TestToolExecutionError:
    """Tests for ToolExecutionError exception class."""

    def test__tool_execution_error__preserves_tool_name(self) -> None:
        """ToolExecutionError stores tool name."""
        original = ValueError("Invalid argument")
        error = ToolExecutionError("calculate", original)

        assert error.tool_name == "calculate"

    def test__tool_execution_error__preserves_original_error(self) -> None:
        """ToolExecutionError stores original error."""
        original = ValueError("Invalid argument")
        error = ToolExecutionError("calculate", original)

        assert error.original_error is original
        assert isinstance(error.original_error, ValueError)

    def test__tool_execution_error__formats_message_correctly(self) -> None:
        """Error message follows consistent format."""
        error = ToolExecutionError("get_weather", KeyError("location"))

        assert "get_weather" in str(error)
        assert "location" in str(error)
        assert str(error) == "Error executing get_weather: 'location'"

    def test__tool_execution_error__works_with_various_exception_types(self) -> None:
        """ToolExecutionError works with different exception types."""
        # TypeError
        error1 = ToolExecutionError("add", TypeError("Expected int"))
        assert "add" in str(error1)
        assert "Expected int" in str(error1)

        # AttributeError
        error2 = ToolExecutionError("fetch", AttributeError("has no attribute 'url'"))
        assert "fetch" in str(error2)
        assert "url" in str(error2)

        # KeyError
        error3 = ToolExecutionError("lookup", KeyError("missing_key"))
        assert "lookup" in str(error3)

    def test__tool_execution_error__inherits_from_exception(self) -> None:
        """ToolExecutionError inherits from Exception."""
        error = ToolExecutionError("test", ValueError("test"))

        assert isinstance(error, Exception)

    def test__tool_execution_error__can_be_raised_and_caught(self) -> None:
        """ToolExecutionError can be raised and caught properly."""
        original = ValueError("test error")

        with pytest.raises(ToolExecutionError) as exc_info:
            raise ToolExecutionError("my_tool", original)

        assert exc_info.value.tool_name == "my_tool"
        assert exc_info.value.original_error is original

    def test__tool_execution_error__exception_chaining_preserved(self) -> None:
        """Original exception preserved in exception chain."""
        original = TypeError("Expected int, got str")

        try:
            raise ToolExecutionError("add", original) from original
        except ToolExecutionError as e:
            assert e.__cause__ is original
            assert isinstance(e.__cause__, TypeError)

    def test__tool_execution_error__with_empty_error_message(self) -> None:
        """ToolExecutionError handles exceptions with empty messages."""
        original = ValueError()
        error = ToolExecutionError("empty_tool", original)

        assert error.tool_name == "empty_tool"
        assert "empty_tool" in str(error)
