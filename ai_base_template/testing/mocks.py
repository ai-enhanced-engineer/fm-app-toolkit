"""Mock LLM implementations for testing LlamaIndex applications."""

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

    This is useful for testing ReAct agents where you need to control
    the exact sequence of thoughts and actions.
    """

    message_chain: list[ChatMessage] = Field(default_factory=list)

    def __init__(self, chain: list[str], **kwargs: Any) -> None:
        """Initialize with a chain of response strings.

        Args:
            chain: List of strings to return as assistant messages
            **kwargs: Additional arguments for the base LLM
        """
        super().__init__(**kwargs)
        self.message_chain = [ChatMessage(role=MessageRole.ASSISTANT, content=message) for message in chain]
        self._current_index = 0

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """Stream the next message in the chain."""

        def gen() -> ChatResponseGen:
            if self._current_index < len(self.message_chain):
                chat_message = self.message_chain[self._current_index]
                self._current_index += 1
                # Stream character by character for more realistic streaming
                content = chat_message.content or ""
                cumulative = ""
                for char in content:
                    cumulative += char
                    yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=cumulative), delta=char)
            else:
                # Return empty response if we've exhausted the chain
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""), delta="")

        return gen()

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        """Async stream the next message in the chain."""

        async def gen() -> ChatResponseAsyncGen:
            if self._current_index < len(self.message_chain):
                chat_message = self.message_chain[self._current_index]
                self._current_index += 1
                # Stream character by character
                content = chat_message.content or ""
                cumulative = ""
                for char in content:
                    cumulative += char
                    yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=cumulative), delta=char)
            else:
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""), delta="")

        return gen()

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Return the next message in the chain."""
        if self._current_index < len(self.message_chain):
            response = self.message_chain[self._current_index]
            self._current_index += 1
            return ChatResponse(message=response)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""))

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Async return the next message in the chain."""
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
        """Reset the index to replay the chain from the beginning."""
        self._current_index = 0


CHUNK_SIZE = 7


class MockLLMEchoStream(LLM):
    """Mock LLM that echoes user input back in streaming chunks.

    This is useful for testing streaming behavior and ensuring
    your application correctly handles streaming responses.
    """

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """Stream the user's message back in chunks."""

        def gen() -> ChatResponseGen:
            # Get the most recent user message
            user_messages = [message.content or "" for message in messages if message.role == MessageRole.USER]

            if not user_messages:
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""), delta="")
                return

            full_content = user_messages[-1]
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

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        """Async stream the user's message back in chunks."""
        user_messages = [message.content or "" for message in messages if message.role == MessageRole.USER]

        if not user_messages:

            async def empty_gen() -> ChatResponseAsyncGen:
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""), delta="")

            return empty_gen()

        full_content = user_messages[-1]

        async def gen() -> ChatResponseAsyncGen:
            cumulative_content = ""
            start_idx = 0

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
        user_messages = [message for message in messages if message.role == MessageRole.USER]

        if user_messages:
            content = user_messages[-1].content or ""
            return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=content))

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
