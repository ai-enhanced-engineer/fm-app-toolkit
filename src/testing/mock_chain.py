"""Mock LLM that returns responses from a predefined sequence for deterministic testing."""

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


class TrajectoryMockLLMLlamaIndex(LLM):
    """Returns responses from a predefined sequence, advancing with each call.

    Usage:
        >>> mock = TrajectoryMockLLMLlamaIndex(chain=["Response 1", "Response 2"])
        >>> response1 = mock.chat([...])  # Gets "Response 1"
        >>> response2 = mock.chat([...])  # Gets "Response 2"
        >>> mock.reset()  # Replay from beginning
    """

    message_chain: list[ChatMessage] = Field(default_factory=list)

    def __init__(self, chain: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Convert string responses to ChatMessage objects
        self.message_chain = [ChatMessage(role=MessageRole.ASSISTANT, content=message) for message in chain]
        # Track position in the chain
        self._current_index = 0

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
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
        # Check if we have messages left in the chain
        if self._current_index < len(self.message_chain):
            response = self.message_chain[self._current_index]
            self._current_index += 1  # Move to next message for next call
            return ChatResponse(message=response)
        # Chain exhausted - return empty response
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
            model_name="mock-llm-chain",
        )

    def reset(self) -> None:
        """Reset to replay the chain from the beginning."""
        self._current_index = 0
