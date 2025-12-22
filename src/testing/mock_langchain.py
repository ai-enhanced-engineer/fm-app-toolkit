"""Mock LangChain chat model for deterministic testing of LangGraph agents.

This module provides a fake chat model that returns responses from a predefined
sequence, enabling deterministic testing without real API calls.
"""

import json
import re
from typing import Any, AsyncIterator, Iterator, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class TrajectoryMockLLMLangChain(BaseChatModel):
    """Returns responses from a predefined sequence for testing LangGraph agents.

    This mock supports tool calling by parsing Action/Action Input patterns
    from the response text and converting them to LangChain tool call format.

    Usage:
        >>> mock = TrajectoryMockLLMLangChain(chain=[
        ...     'Thought: Need weather\nAction: get_weather\nAction Input: {"location": "Tokyo"}',
        ...     'Thought: Got it\nAnswer: Weather in Tokyo: 75°F and sunny'
        ... ])
        >>> result = await mock.ainvoke([HumanMessage(content="What's the weather?")])
        >>> # First call has tool_calls, second has final answer
    """

    chain: list[str] = Field(default_factory=list)
    _current_index: int = 0

    def __init__(self, chain: list[str], **kwargs: Any) -> None:
        """Initialize with a sequence of responses.

        Args:
            chain: List of response strings to return in sequence
            **kwargs: Additional arguments for BaseChatModel
        """
        super().__init__(chain=chain, **kwargs)
        self._current_index = 0

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "mock-chat-model-chain"

    def _parse_tool_call(self, text: str) -> tuple[str | None, dict[str, Any]]:
        """Parse Action and Action Input from ReAct-style text.

        Args:
            text: Response text to parse

        Returns:
            Tuple of (action_name, action_input_dict)
        """
        # Extract action name
        action_match = re.search(r"Action:\s*(\w+)", text)
        action = action_match.group(1) if action_match else None

        # Extract action input (JSON format)
        action_input: dict[str, Any] = {}
        if action:
            input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
            if input_match:
                try:
                    action_input = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    action_input = {}

        return action, action_input

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: Sequence[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the chain.

        Args:
            messages: Input messages (ignored, uses chain sequence)
            stop: Stop sequences (ignored)
            run_manager: Callback manager (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            ChatResult with the next message in the chain
        """
        # Get next response from chain
        if self._current_index < len(self.chain):
            text = self.chain[self._current_index]
            self._current_index += 1
        else:
            # Chain exhausted
            text = "I don't have a response for this."

        # Parse for tool calls
        action, action_input = self._parse_tool_call(text)

        # Create AI message
        if action:
            # Response includes a tool call
            message = AIMessage(
                content=text,
                tool_calls=[
                    {
                        "name": action,
                        "args": action_input,
                        "id": f"call_{self._current_index}",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            # Regular text response
            message = AIMessage(content=text)

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: Sequence[BaseMessage],
        stop: Sequence[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of _generate.

        Args:
            messages: Input messages (ignored, uses chain sequence)
            stop: Stop sequences (ignored)
            run_manager: Callback manager (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            ChatResult with the next message in the chain
        """
        return self._generate(messages, stop, run_manager, **kwargs)

    def _stream(
        self,
        messages: Sequence[BaseMessage],
        stop: Sequence[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream response (not implemented for this mock)."""
        raise NotImplementedError("Streaming not supported in mock")

    def _astream(
        self,
        messages: Sequence[BaseMessage],
        stop: Sequence[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream response (not implemented for this mock)."""
        raise NotImplementedError("Streaming not supported in mock")

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> "TrajectoryMockLLMLangChain":
        """Bind tools to the model (no-op for mock, returns self).

        Args:
            tools: Sequence of tools to bind (ignored for mock)
            **kwargs: Additional arguments (ignored)

        Returns:
            Self for method chaining
        """
        # For the mock, we don't need to do anything with tools
        # The tool calls are already embedded in the chain responses
        return self

    def reset(self) -> None:
        """Reset to replay the chain from the beginning."""
        self._current_index = 0
