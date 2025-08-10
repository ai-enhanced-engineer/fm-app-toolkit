"""Rule-based mock LLM for behavior-driven testing.

This module provides RuleBasedMockLLM, a mock that generates contextual
responses based on rules and input content rather than predefined scripts.
This enables more realistic and valuable behavior-based testing.

Key Features:
- Dynamic response generation based on input content
- Configurable rules for specific keywords/patterns
- Multiple default behaviors (direct_answer, use_tool, cannot_answer)
- Context-aware tool selection
- Call counting for interaction tracking

Example Usage:
    Testing dynamic agent behavior:

    >>> rules = {
    ...     "weather": "Thought: Check weather.\\nAction: get_weather\\nAction Input: {{}}",
    ...     "calculate": "Thought: Do math.\\nAction: calculate\\nAction Input: {{}}",
    ...     "hello": "Thought: Greeting detected.\\nAnswer: Hello! How can I help?"
    ... }
    >>>
    >>> mock = RuleBasedMockLLM(rules=rules, default_behavior="use_tool")
    >>>
    >>> # Mock responds based on content, not a fixed sequence
    >>> response1 = mock.chat([user_message("What's the weather?")])
    >>> # Returns weather rule response
    >>>
    >>> response2 = mock.chat([user_message("Calculate 2+2")])
    >>> # Returns calculate rule response
    >>>
    >>> response3 = mock.chat([user_message("Unknown request")])
    >>> # Falls back to default_behavior

When to Use:
- Testing agent's ability to handle varied inputs
- Validating tool selection logic
- Testing fallback behaviors
- More realistic than scripted responses
- Testing agent adaptation to different queries
"""

from typing import Any, Dict, Optional, Sequence

from llama_cloud import MessageRole
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.llms.llm import LLM
from pydantic import Field


class RuleBasedMockLLM(LLM):
    """A mock LLM that generates contextual responses based on rules.

    Unlike MockLLMWithChain which returns predefined responses, this mock
    generates responses based on the input and available context, making
    tests more realistic and valuable.

    This enables behavior-based testing where the mock intelligently responds
    to queries rather than following a script, making tests more robust and
    educational.

    How It Works:
        1. Examines the user's message for keywords
        2. Checks if any rules match those keywords
        3. If a rule matches, uses the rule's response pattern
        4. If no rules match, falls back to default behavior
        5. Default behavior can be context-aware (e.g., selecting tools)

    Response Patterns:
        Rules can include placeholders that get filled:
        - {query}: The original user query
        - {{}}: Empty JSON object for action input

    Default Behaviors:
        - "direct_answer": Provides a direct response
        - "use_tool": Intelligently selects a tool based on query
        - "cannot_answer": Indicates inability to help
        - Any other value: Generic processing response

    Example:
        Testing with specific rules:

        >>> rules = {
        ...     "time": "Thought: Get current time.\\nAction: get_time\\nAction Input: {{}}",
        ...     "joke": "Thought: Tell a joke.\\nAnswer: Why did the test pass? It asserted itself!",
        ... }
        >>> mock = RuleBasedMockLLM(rules=rules, default_behavior="cannot_answer")
        >>>
        >>> # Matches "time" rule
        >>> response = mock.chat([user_message("What time is it?")])
        >>> assert "get_time" in response.message.content
        >>>
        >>> # Matches "joke" rule
        >>> response = mock.chat([user_message("Tell me a joke")])
        >>> assert "asserted itself" in response.message.content
        >>>
        >>> # No rule matches, uses default
        >>> response = mock.chat([user_message("Random question")])
        >>> assert "cannot answer" in response.message.content.lower()

        Testing with intelligent tool selection:

        >>> mock = RuleBasedMockLLM(rules={}, default_behavior="use_tool")
        >>>
        >>> # Detects "add" keyword and selects add tool
        >>> response = mock.chat([user_message("Please add 5 and 3")])
        >>> assert "Action: add" in response.message.content
        >>>
        >>> # Detects "weather" and selects weather tool
        >>> response = mock.chat([user_message("What's the weather?")])
        >>> assert "Action: get_weather" in response.message.content

    Attributes:
        rules: Dictionary mapping keywords to response patterns
        default_behavior: Fallback behavior when no rules match
        call_count: Tracks number of chat calls for testing
    """

    rules: Dict[str, str] = Field(default_factory=dict)
    default_behavior: str = Field(default="direct_answer")
    call_count: int = Field(default=0)

    def __init__(
        self, rules: Optional[Dict[str, str]] = None, default_behavior: str = "direct_answer", **kwargs: Any
    ) -> None:
        """Initialize the rule-based mock.

        Args:
            rules: Dictionary mapping keywords to response patterns.
                   Keys are keywords to search for (case-insensitive).
                   Values are response templates with optional {query} placeholder.
            default_behavior: What to do when no rules match.
                            Options: "direct_answer", "use_tool", "cannot_answer", or custom.
            **kwargs: Additional arguments for the base LLM class.

        Example:
            >>> rules = {
            ...     "search": "Thought: Searching.\\nAction: search\\nAction Input: {{'q': '{query}'}}",
            ...     "help": "Thought: User needs help.\\nAnswer: I'm here to assist you!",
            ... }
            >>> mock = RuleBasedMockLLM(rules=rules, default_behavior="use_tool")
        """
        super().__init__(**kwargs)
        self.rules = rules or {}
        self.default_behavior = default_behavior
        self.call_count = 0

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Generate a response based on rules and context.

        Examines the input messages and generates an appropriate response
        based on configured rules and default behavior.

        Args:
            messages: List of chat messages to process
            **kwargs: Additional arguments (ignored)

        Returns:
            ChatResponse with generated content based on rules
        """
        self.call_count += 1

        # Extract the last user message
        user_msg = ""
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                user_msg = msg.content or ""
                break

        # Check for tool-related context in system message
        has_tools = any("tool" in (msg.content or "").lower() for msg in messages if msg.role == MessageRole.SYSTEM)

        # Apply rules based on content
        response = self._apply_rules(user_msg, has_tools)

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response),
            raw={},
        )

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Async version of chat.

        Simply delegates to sync version since no real I/O is involved.

        Args:
            messages: List of chat messages to process
            **kwargs: Additional arguments (ignored)

        Returns:
            ChatResponse with generated content
        """
        return self.chat(messages, **kwargs)

    def _apply_rules(self, user_msg: str, has_tools: bool) -> str:
        """Apply rules to generate response.

        Internal method that implements the rule matching and response generation logic.

        Args:
            user_msg: The user's message content
            has_tools: Whether tools are available in the context

        Returns:
            Generated response string based on rules or default behavior
        """
        user_msg_lower = user_msg.lower()

        # Check each rule for keyword matches
        for keyword, pattern in self.rules.items():
            if keyword.lower() in user_msg_lower:
                # Use format to replace {query} placeholder if present
                return pattern.format(query=user_msg)

        # No rule matched, use default behavior
        if self.default_behavior == "use_tool" and has_tools:
            # Intelligently select a tool based on query keywords
            if "add" in user_msg_lower or "sum" in user_msg_lower or "plus" in user_msg_lower:
                return "Thought: I need to add numbers.\\nAction: add\\nAction Input: {}"
            elif "multiply" in user_msg_lower or "times" in user_msg_lower:
                return "Thought: I need to multiply numbers.\\nAction: multiply\\nAction Input: {}"
            elif "weather" in user_msg_lower:
                return "Thought: I need to check the weather.\\nAction: get_weather\\nAction Input: {}"
            else:
                return "Thought: I'll try to help with this.\\nAnswer: I'll do my best to help."

        elif self.default_behavior == "direct_answer":
            return f"Thought: I can answer directly.\\nAnswer: Response to: {user_msg}"

        elif self.default_behavior == "cannot_answer":
            return "Thought: I cannot answer this with available tools.\\nAnswer: I'm sorry, I cannot help with that."

        else:
            # Custom default behavior
            return f"Thought: Processing request.\\nAnswer: Processed: {user_msg}"

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """Stream the response character by character.

        Generates response using rules then streams it character by character
        for testing streaming behavior.

        Args:
            messages: List of chat messages to process
            **kwargs: Additional arguments (ignored)

        Yields:
            ChatResponse objects with cumulative content and character deltas
        """

        def gen() -> ChatResponseGen:
            response = self.chat(messages, **kwargs)
            content = response.message.content or ""
            cumulative = ""
            for char in content:
                cumulative += char
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=cumulative), delta=char)

        return gen()

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        """Async stream the response character by character.

        Async version of stream_chat for testing async streaming workflows.

        Args:
            messages: List of chat messages to process
            **kwargs: Additional arguments (ignored)

        Yields:
            ChatResponse objects with cumulative content and character deltas
        """

        async def gen() -> ChatResponseAsyncGen:
            response = await self.achat(messages, **kwargs)
            content = response.message.content or ""
            cumulative = ""
            for char in content:
                cumulative += char
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=cumulative), delta=char)

        return gen()

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        """Not implemented for this mock."""
        raise NotImplementedError("Use chat methods for this mock")

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        """Not implemented for this mock."""
        raise NotImplementedError("Use chat methods for this mock")

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        """Not implemented for this mock."""
        raise NotImplementedError("Use chat methods for this mock")

    def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        """Not implemented for this mock."""
        raise NotImplementedError("Use chat methods for this mock")

    @property
    def metadata(self) -> LLMMetadata:
        """Return mock metadata.

        Returns metadata indicating this is a chat model for testing.
        """
        return LLMMetadata(
            context_window=4096,
            num_output=256,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name="mock-llm-rule-based",
        )
