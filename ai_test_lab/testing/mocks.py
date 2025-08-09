"""Mock LLM implementations for testing LlamaIndex applications.

This module provides deterministic mock LLMs that simulate real LLM behavior
for testing purposes, enabling fast, cost-free, and reproducible tests.
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
from pydantic import Field


class MockLLMWithChain(LLM):
    """Mock LLM that returns a predefined sequence of responses.
    
    This mock simulates a real LLM by returning responses from a predefined chain
    in sequential order. Each call to chat() or stream_chat() advances through
    the chain, making it perfect for testing multi-step agent workflows where
    you need deterministic, reproducible behavior.
    
    Example:
        >>> mock_llm = MockLLMWithChain(chain=[
        ...     "Thought: I need to search.\nAction: search\nAction Input: {'q': 'test'}",
        ...     "Thought: Found it.\nAnswer: The answer is 42"
        ... ])
        >>> # First call returns first response
        >>> response1 = mock_llm.chat([...])  # Returns first chain element
        >>> # Second call returns second response  
        >>> response2 = mock_llm.chat([...])  # Returns second chain element
        >>> # Third call returns empty (chain exhausted)
        >>> response3 = mock_llm.chat([...])  # Returns empty response
    
    The chain can be reset to replay from the beginning using reset().
    """

    message_chain: list[ChatMessage] = Field(default_factory=list)

    def __init__(self, chain: list[str], **kwargs: Any) -> None:
        """Initialize with a chain of response strings.

        Args:
            chain: List of strings to return as assistant messages
            **kwargs: Additional arguments for the base LLM
        """
        super().__init__(**kwargs)
        # Convert string responses to ChatMessage objects
        self.message_chain = [ChatMessage(role=MessageRole.ASSISTANT, content=message) for message in chain]
        # Track position in the chain
        self._current_index = 0

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """Stream the next message in the chain character by character."""

        def gen() -> ChatResponseGen:
            # Check if we still have messages in the chain
            if self._current_index < len(self.message_chain):
                chat_message = self.message_chain[self._current_index]
                self._current_index += 1  # Advance to next message for next call
                
                # Stream character by character for realistic streaming behavior
                content = chat_message.content or ""
                cumulative = ""
                for char in content:
                    cumulative += char
                    yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=cumulative), delta=char)
            else:
                # Chain exhausted - return empty response
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""), delta="")

        return gen()

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        """Async stream the next message in the chain character by character."""

        async def gen() -> ChatResponseAsyncGen:
            # Check if we still have messages in the chain
            if self._current_index < len(self.message_chain):
                chat_message = self.message_chain[self._current_index]
                self._current_index += 1  # Advance for next call
                
                # Stream character by character
                content = chat_message.content or ""
                cumulative = ""
                for char in content:
                    cumulative += char
                    yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=cumulative), delta=char)
            else:
                # Chain exhausted
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""), delta="")

        return gen()

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Return the next message in the chain."""
        # Check if we have messages left in the chain
        if self._current_index < len(self.message_chain):
            response = self.message_chain[self._current_index]
            self._current_index += 1  # Move to next message for next call
            return ChatResponse(message=response)
        # Chain exhausted - return empty response
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""))

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Async return the next message in the chain."""
        # Simply delegate to sync version since we're not doing real I/O
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
        """Return mock metadata."""
        return LLMMetadata(
            context_window=4096,
            num_output=256,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name="mock-llm-chain",
        )

    def reset(self) -> None:
        """Reset the index to replay the chain from the beginning.
        
        This allows you to reuse the same mock instance for multiple test runs
        or to test error recovery scenarios where the agent might retry.
        """
        self._current_index = 0


CHUNK_SIZE = 7


class MockLLMEchoStream(LLM):
    """Mock LLM that echoes user input back in streaming chunks.
    
    This mock is useful for testing streaming behavior and ensuring
    your application correctly handles streaming responses. It takes
    the most recent user message and echoes it back in chunks.
    
    Example:
        >>> mock_llm = MockLLMEchoStream()
        >>> response = mock_llm.chat([
        ...     ChatMessage(role=MessageRole.USER, content="Hello world")
        ... ])
        >>> response.message.content  # "Hello world"
    
    The streaming version chunks the response into CHUNK_SIZE characters.
    """

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """Stream the user's message back in chunks."""

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
        """Async stream the user's message back in chunks."""
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
        """Echo the most recent user message."""
        # Find the last user message
        user_messages = [message for message in messages if message.role == MessageRole.USER]

        if user_messages:
            content = user_messages[-1].content or ""
            return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=content))

        # No user message - return empty
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""))

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Async echo the most recent user message."""
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
        """Return mock metadata."""
        return LLMMetadata(
            context_window=4096,
            num_output=256,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name="mock-llm-echo",
        )