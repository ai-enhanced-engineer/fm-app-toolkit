"""Tests for common agent types."""

import pytest

from src.agents.common.types import Tool


class TestToolDataclass:
    """Tests for the Tool dataclass."""

    def test__tool_dataclass__creates_valid_instance(self) -> None:
        """Create Tool with all required fields."""

        def sample_fn(x: int) -> int:
            return x * 2

        tool = Tool(name="multiply", description="Multiply by 2", function=sample_fn)

        assert tool.name == "multiply"
        assert tool.description == "Multiply by 2"
        assert tool.function(5) == 10

    def test__tool_dataclass__missing_name_raises_type_error(self) -> None:
        """Tool requires name field."""
        with pytest.raises(TypeError):
            Tool(description="desc", function=lambda x: x)  # type: ignore[call-arg]

    def test__tool_dataclass__missing_description_raises_type_error(self) -> None:
        """Tool requires description field."""
        with pytest.raises(TypeError):
            Tool(name="test", function=lambda x: x)  # type: ignore[call-arg]

    def test__tool_dataclass__missing_function_raises_type_error(self) -> None:
        """Tool requires function field."""
        with pytest.raises(TypeError):
            Tool(name="test", description="desc")  # type: ignore[call-arg]

    def test__tool_dataclass__accepts_lambda(self) -> None:
        """Tool accepts lambda as function."""
        tool = Tool(name="lambda", description="Lambda fn", function=lambda x: x + 1)
        assert tool.function(5) == 6

    def test__tool_dataclass__accepts_class_method(self) -> None:
        """Tool accepts class method as function."""

        class Calculator:
            def add(self, a: int, b: int) -> int:
                return a + b

        calc = Calculator()
        tool = Tool(name="add", description="Add numbers", function=calc.add)
        assert tool.function(2, 3) == 5

    def test__tool_dataclass__accepts_async_function(self) -> None:
        """Tool accepts async function (stored but not awaited in test)."""

        async def async_fn(x: int) -> int:
            return x * 2

        tool = Tool(name="async", description="Async function", function=async_fn)
        assert tool.function is async_fn

    def test__tool_dataclass__preserves_function_kwargs(self) -> None:
        """Tool function can be called with keyword arguments."""

        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        tool = Tool(name="greet", description="Greet someone", function=greet)
        assert tool.function(name="World") == "Hello, World!"
        assert tool.function(name="World", greeting="Hi") == "Hi, World!"
