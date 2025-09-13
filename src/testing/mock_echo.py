"""Mock LLM that echoes user input back in chunks for testing streaming behavior."""

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
# Using 7 to create multiple chunks for testing streaming behavior
CHUNK_SIZE = 7


class MockLLMEchoStream(LLM):
    """Echoes the last user message back, optionally in chunks for streaming tests.

    Usage:
        >>> mock = MockLLMEchoStream()
        >>> response = mock.chat([ChatMessage(role=MessageRole.USER, content="Hi")])
        >>> assert response.message.content == "Hi"
    """

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
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
        # Find the last user message
        user_messages = [message for message in messages if message.role == MessageRole.USER]

        if user_messages:
            content = user_messages[-1].content or ""
            return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=content))

        # No user message - return empty
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""))

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise NotImplementedError("Use chat methods for this mock")

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise NotImplementedError("Use chat methods for this mock")

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise NotImplementedError("Use chat methods for this mock")

    def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise NotImplementedError("Use chat methods for this mock")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=256,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name="mock-llm-echo",
        )
