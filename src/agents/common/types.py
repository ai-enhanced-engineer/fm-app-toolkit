"""Shared types for agent implementations.

This module provides common type definitions used across different agent
frameworks (LangGraph, LlamaIndex, etc.) to eliminate duplication.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    """Simple tool representation - a function with metadata.

    A unified tool definition that can be converted to framework-specific
    tool formats (LangChain tools, LlamaIndex FunctionTools, etc.).

    Attributes:
        name: The unique identifier for the tool.
        description: Human-readable description of what the tool does.
        function: The callable that implements the tool's functionality.

    Example:
        >>> def calculate(expression: str) -> str:
        ...     return str(eval(expression))
        >>> tool = Tool(name="calculate", description="Evaluate math", function=calculate)
    """

    name: str
    description: str
    function: Callable[..., Any]
