"""Mock LLM that echoes user input for testing streaming behavior.

This module provides MockLLMEchoStream, a mock LLM that echoes the user's
input back in configurable chunks. Perfect for testing streaming behavior,
chunk processing, and ensuring your application correctly handles streaming
responses.

Key Features:
- Echoes the last user message back to test input processing
- Chunks responses for streaming tests (default: 7 characters)
- Validates cumulative content building in streaming
- Both sync and async streaming support

Example Usage:
    Testing streaming chunk processing:

    >>> mock_llm = MockLLMEchoStream()
    >>> messages = [
    ...     ChatMessage(role=MessageRole.USER, content="Hello, world!")
    ... ]
    >>>
    >>> # Streaming returns chunks: "Hello, ", "world!"
    >>> async for chunk in mock_llm.astream_chat(messages):
    ...     print(f"Delta: '{chunk.delta}'")
    ...     print(f"Cumulative: '{chunk.message.content}'")

    Output:
    Delta: 'Hello, '
    Cumulative: 'Hello, '
    Delta: 'world!'
    Cumulative: 'Hello, world!'

When to Use:
- Testing streaming response handling
- Validating chunk processing logic
- Testing cumulative content building
- Ensuring UI updates correctly with streaming
- Testing async streaming workflows
"""

from typing import Any, Sequence

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

# Default chunk size for streaming responses
CHUNK_SIZE = 7


class MockLLMEchoStream(LLM):
    """Mock LLM that echoes user input back in streaming chunks.

    This mock is essential for testing streaming behavior and ensuring
    your application correctly handles streaming responses. It takes
    the most recent user message and echoes it back, optionally in
    chunks for streaming methods.

    The echo behavior makes it easy to verify that:
    1. User input is correctly extracted from messages
    2. Streaming chunks are processed in order
    3. Cumulative content builds correctly
    4. Final content matches the input

    Streaming Behavior:
        The mock chunks responses into CHUNK_SIZE characters (default: 7).
        This helps test:
        - Partial token processing
        - UI updates with incomplete responses
        - Buffer handling in streaming
        - Progress indicators

    Example:
        Basic echo test:

        >>> mock_llm = MockLLMEchoStream()
        >>> response = mock_llm.chat([
        ...     ChatMessage(role=MessageRole.USER, content="Test message")
        ... ])
        >>> assert response.message.content == "Test message"

        Streaming test:

        >>> # Input: "Hello world" (11 chars)
        >>> # Chunks with CHUNK_SIZE=7: "Hello w", "orld"
        >>> chunks = []
        >>> for chunk in mock_llm.stream_chat(messages):
        ...     chunks.append(chunk.delta)
        >>> assert chunks == ["Hello w", "orld"]

        Testing with no user message:

        >>> response = mock_llm.chat([
        ...     ChatMessage(role=MessageRole.SYSTEM, content="System prompt")
        ... ])
        >>> assert response.message.content == ""  # No user message to echo

    Note:
        Only the LAST user message is echoed. This simulates how LLMs
        typically respond to the most recent user input.
    """

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """Stream the user's message back in chunks.

        Extracts the last user message and streams it back in CHUNK_SIZE pieces.
        This helps test streaming response handling in your application.

        Args:
            messages: List of chat messages, will echo the last USER message
            **kwargs: Additional arguments (ignored)

        Yields:
            ChatResponse objects with cumulative content and chunk deltas
        """

        def gen() -> ChatResponseGen:
            # Get the most recent user message to echo
            user_messages = [message.content or "" for message in messages if message.role == MessageRole.USER]

            if not user_messages:
                # No user message to echo - return empty
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""), delta="")
                return

            full_content = user_messages[-1]  # Echo the last user message
            cumulative_content = ""
            start_idx = 0

            # Stream in chunks of CHUNK_SIZE
            while start_idx < len(full_content):
                chunk = full_content[start_idx : start_idx + CHUNK_SIZE]
                cumulative_content += chunk
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=cumulative_content),
                    delta=chunk,
                )
                start_idx += CHUNK_SIZE

        return gen()

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        """Async stream the user's message back in chunks.

        Async version of stream_chat for testing async streaming workflows.

        Args:
            messages: List of chat messages, will echo the last USER message
            **kwargs: Additional arguments (ignored)

        Yields:
            ChatResponse objects with cumulative content and chunk deltas
        """
        user_messages = [message.content or "" for message in messages if message.role == MessageRole.USER]

        if not user_messages:
            # No user message to echo
            async def empty_gen() -> ChatResponseAsyncGen:
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""), delta="")

            return empty_gen()

        full_content = user_messages[-1]  # Echo the last user message

        async def gen() -> ChatResponseAsyncGen:
            cumulative_content = ""
            start_idx = 0

            # Stream in chunks
            while start_idx < len(full_content):
                chunk = full_content[start_idx : start_idx + CHUNK_SIZE]
                cumulative_content += chunk
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=cumulative_content),
                    delta=chunk,
                )
                start_idx += CHUNK_SIZE

        return gen()

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Echo the most recent user message.

        Non-streaming version that returns the complete echo immediately.

        Args:
            messages: List of chat messages, will echo the last USER message
            **kwargs: Additional arguments (ignored)

        Returns:
            ChatResponse with the echoed user message or empty if no user message
        """
        # Find the last user message
        user_messages = [message for message in messages if message.role == MessageRole.USER]

        if user_messages:
            content = user_messages[-1].content or ""
            return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=content))

        # No user message - return empty
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""))

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Async echo the most recent user message.

        Simply delegates to sync version since no real I/O is involved.

        Args:
            messages: List of chat messages, will echo the last USER message
            **kwargs: Additional arguments (ignored)

        Returns:
            ChatResponse with the echoed user message
        """
        return self.chat(messages, **kwargs)

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
            model_name="mock-llm-echo",
        )
