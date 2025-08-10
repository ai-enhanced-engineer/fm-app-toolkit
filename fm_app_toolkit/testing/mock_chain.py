"""Mock LLM with predefined chain of responses for deterministic testing.

This module provides MockLLMWithChain, a mock LLM that returns responses from
a predefined sequence. Perfect for testing multi-step agent workflows where
you need deterministic, reproducible behavior.

Key Features:
- Sequential response delivery from a predefined chain
- Character-by-character streaming for realistic behavior
- Reset capability to replay sequences
- Both sync and async support

Example Usage:
    Testing a multi-step reasoning chain:
    
    >>> chain = [
    ...     "Thought: I need to search for information.\\n"
    ...     "Action: search\\n"
    ...     "Action Input: {'query': 'LlamaIndex testing'}",
    ...     
    ...     "Thought: Found the information, now I'll summarize.\\n"
    ...     "Answer: LlamaIndex provides testing utilities..."
    ... ]
    >>> mock_llm = MockLLMWithChain(chain=chain)
    >>> agent = MyAgent(llm=mock_llm)
    >>> result = await agent.run("How do I test LlamaIndex?")
    
    The mock will return each response in sequence, allowing you to test
    the agent's handling of multi-step workflows deterministically.

When to Use:
- Testing specific agent workflows with known steps
- Validating error handling with controlled responses
- Ensuring reproducible test results
- Testing parsing of specific response formats
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
    
    The chain represents a conversation flow where each element is what the
    assistant would respond at that step. This is particularly useful for:
    
    1. Testing ReAct agents with specific reasoning patterns
    2. Validating error recovery (by including error responses)
    3. Testing max iteration limits (by providing exact number of responses)
    4. Ensuring consistent test results across runs
    
    Attributes:
        message_chain: List of ChatMessage objects created from input strings
        _current_index: Tracks position in the chain (not exposed publicly)
    
    Example:
        Testing a calculation workflow:
        
        >>> mock_llm = MockLLMWithChain(chain=[
        ...     "Thought: I need to multiply first.\\n"
        ...     "Action: multiply\\n"
        ...     "Action Input: {'a': 3, 'b': 4}",
        ...     
        ...     "Thought: 3 * 4 = 12. Now I'll add 5.\\n"
        ...     "Action: add\\n"
        ...     "Action Input: {'a': 12, 'b': 5}",
        ...     
        ...     "Thought: The final result is 17.\\n"
        ...     "Answer: (3 × 4) + 5 = 17"
        ... ])
        >>> 
        >>> # First call gets multiplication step
        >>> response1 = mock_llm.chat([...])  
        >>> # Second call gets addition step
        >>> response2 = mock_llm.chat([...])  
        >>> # Third call gets final answer
        >>> response3 = mock_llm.chat([...])
        >>> 
        >>> # Reset to replay from beginning
        >>> mock_llm.reset()
        >>> response1_again = mock_llm.chat([...])  # Back to first response
    
    Note:
        Once the chain is exhausted, subsequent calls return empty responses.
        Use reset() to replay the chain from the beginning.
    """

    message_chain: list[ChatMessage] = Field(default_factory=list)

    def __init__(self, chain: list[str], **kwargs: Any) -> None:
        """Initialize with a chain of response strings.

        Args:
            chain: List of strings to return as assistant messages.
                   Each string represents one complete response from the assistant.
            **kwargs: Additional arguments for the base LLM class.
        
        Example:
            >>> chain = [
            ...     "First response from assistant",
            ...     "Second response from assistant",
            ...     "Third response from assistant"
            ... ]
            >>> mock = MockLLMWithChain(chain=chain)
        """
        super().__init__(**kwargs)
        # Convert string responses to ChatMessage objects
        self.message_chain = [
            ChatMessage(role=MessageRole.ASSISTANT, content=message) 
            for message in chain
        ]
        # Track position in the chain
        self._current_index = 0

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """Stream the next message in the chain character by character.
        
        This simulates realistic streaming behavior by yielding one character
        at a time from the current response in the chain.
        
        Args:
            messages: Input messages (ignored - we use the chain)
            **kwargs: Additional arguments (ignored)
            
        Yields:
            ChatResponse objects with cumulative content and character deltas
        """
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
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=cumulative), 
                        delta=char
                    )
            else:
                # Chain exhausted - return empty response
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=""), 
                    delta=""
                )

        return gen()

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        """Async stream the next message in the chain character by character.
        
        Async version of stream_chat for compatibility with async workflows.
        
        Args:
            messages: Input messages (ignored - we use the chain)
            **kwargs: Additional arguments (ignored)
            
        Yields:
            ChatResponse objects with cumulative content and character deltas
        """
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
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=cumulative), 
                        delta=char
                    )
            else:
                # Chain exhausted
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=""), 
                    delta=""
                )

        return gen()

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Return the next message in the chain.
        
        Args:
            messages: Input messages (ignored - we use the chain)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            ChatResponse with the next message in the chain, or empty if exhausted
        """
        # Check if we have messages left in the chain
        if self._current_index < len(self.message_chain):
            response = self.message_chain[self._current_index]
            self._current_index += 1  # Move to next message for next call
            return ChatResponse(message=response)
        # Chain exhausted - return empty response
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""))

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Async return the next message in the chain.
        
        Simply delegates to sync version since we're not doing real I/O.
        
        Args:
            messages: Input messages (ignored - we use the chain)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            ChatResponse with the next message in the chain
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
            model_name="mock-llm-chain",
        )

    def reset(self) -> None:
        """Reset the index to replay the chain from the beginning.
        
        This allows you to reuse the same mock instance for multiple test runs
        or to test error recovery scenarios where the agent might retry.
        
        Example:
            >>> mock = MockLLMWithChain(chain=["Response 1", "Response 2"])
            >>> response1 = mock.chat([...])  # Gets "Response 1"
            >>> response2 = mock.chat([...])  # Gets "Response 2"
            >>> response3 = mock.chat([...])  # Gets empty (exhausted)
            >>> 
            >>> mock.reset()
            >>> response1_again = mock.chat([...])  # Gets "Response 1" again
        """
        self._current_index = 0