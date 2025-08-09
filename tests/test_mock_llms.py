"""Tests for mock LLM implementations."""

import pytest
from llama_cloud import MessageRole
from llama_index.core.base.llms.types import ChatMessage

from ai_base_template.testing.mocks import MockLLMEchoStream, MockLLMWithChain


class TestMockLLMWithChain:
    """Test the MockLLMWithChain implementation."""

    def test_init_creates_message_chain(self) -> None:
        """Test that initialization creates the correct message chain."""
        chain = ["First response", "Second response", "Third response"]
        mock_llm = MockLLMWithChain(chain=chain)

        assert len(mock_llm.message_chain) == 3
        assert all(msg.role == MessageRole.ASSISTANT for msg in mock_llm.message_chain)
        assert mock_llm.message_chain[0].content == "First response"
        assert mock_llm.message_chain[1].content == "Second response"
        assert mock_llm.message_chain[2].content == "Third response"

    def test_chat_returns_messages_in_sequence(self) -> None:
        """Test that chat returns messages in the correct sequence."""
        chain = ["Response 1", "Response 2", "Response 3"]
        mock_llm = MockLLMWithChain(chain=chain)

        # First call
        response1 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Question 1")])
        assert response1.message.content == "Response 1"

        # Second call
        response2 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Question 2")])
        assert response2.message.content == "Response 2"

        # Third call
        response3 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Question 3")])
        assert response3.message.content == "Response 3"

        # Fourth call (beyond chain length)
        response4 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Question 4")])
        assert response4.message.content == ""

    def test_stream_chat_streams_characters(self) -> None:
        """Test that stream_chat streams response character by character."""
        chain = ["Hello"]
        mock_llm = MockLLMWithChain(chain=chain)

        messages = [ChatMessage(role=MessageRole.USER, content="Hi")]
        stream = mock_llm.stream_chat(messages)

        chunks = list(stream)
        assert len(chunks) == 5  # "Hello" has 5 characters

        # Check deltas
        assert chunks[0].delta == "H"
        assert chunks[1].delta == "e"
        assert chunks[2].delta == "l"
        assert chunks[3].delta == "l"
        assert chunks[4].delta == "o"

        # Check cumulative content
        assert chunks[0].message.content == "H"
        assert chunks[1].message.content == "He"
        assert chunks[2].message.content == "Hel"
        assert chunks[3].message.content == "Hell"
        assert chunks[4].message.content == "Hello"

    @pytest.mark.asyncio
    async def test_astream_chat_async_streaming(self) -> None:
        """Test async streaming of responses."""
        chain = ["Test"]
        mock_llm = MockLLMWithChain(chain=chain)

        messages = [ChatMessage(role=MessageRole.USER, content="Query")]
        stream = await mock_llm.astream_chat(messages)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) == 4  # "Test" has 4 characters
        assert "".join(c.delta for c in chunks) == "Test"

    def test_reset_replays_chain(self) -> None:
        """Test that reset allows replaying the chain."""
        chain = ["First", "Second"]
        mock_llm = MockLLMWithChain(chain=chain)

        # Use up the chain
        response1 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Q1")])
        response2 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Q2")])
        assert response1.message.content == "First"
        assert response2.message.content == "Second"

        # Reset and replay
        mock_llm.reset()
        response3 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Q3")])
        assert response3.message.content == "First"

    def test_metadata_properties(self) -> None:
        """Test that metadata returns expected properties."""
        mock_llm = MockLLMWithChain(chain=["test"])
        metadata = mock_llm.metadata

        assert metadata.context_window == 4096
        assert metadata.num_output == 256
        assert metadata.is_chat_model is True
        assert metadata.is_function_calling_model is False
        assert metadata.model_name == "mock-llm-chain"


class TestMockLLMEchoStream:
    """Test the MockLLMEchoStream implementation."""

    def test_chat_echoes_user_message(self) -> None:
        """Test that chat echoes the most recent user message."""
        mock_llm = MockLLMEchoStream()

        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello, world!"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Previous response"),
            ChatMessage(role=MessageRole.USER, content="Echo this"),
        ]

        response = mock_llm.chat(messages)
        assert response.message.content == "Echo this"

    def test_chat_handles_empty_messages(self) -> None:
        """Test handling of empty message list."""
        mock_llm = MockLLMEchoStream()

        response = mock_llm.chat([])
        assert response.message.content == ""

    def test_stream_chat_chunks_correctly(self) -> None:
        """Test that streaming chunks messages correctly."""
        mock_llm = MockLLMEchoStream()

        # Test with message exactly CHUNK_SIZE
        messages = [ChatMessage(role=MessageRole.USER, content="1234567")]  # 7 chars
        stream = mock_llm.stream_chat(messages)
        chunks = list(stream)

        assert len(chunks) == 1
        assert chunks[0].delta == "1234567"
        assert chunks[0].message.content == "1234567"

        # Test with longer message
        messages = [ChatMessage(role=MessageRole.USER, content="1234567890")]  # 10 chars
        mock_llm = MockLLMEchoStream()  # Reset
        stream = mock_llm.stream_chat(messages)
        chunks = list(stream)

        assert len(chunks) == 2
        assert chunks[0].delta == "1234567"
        assert chunks[1].delta == "890"
        assert chunks[1].message.content == "1234567890"

    @pytest.mark.asyncio
    async def test_astream_chat_echoes_async(self) -> None:
        """Test async streaming echo."""
        mock_llm = MockLLMEchoStream()

        messages = [ChatMessage(role=MessageRole.USER, content="Async test")]
        stream = await mock_llm.astream_chat(messages)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        # "Async test" = 10 chars, should be 2 chunks (7 + 3)
        assert len(chunks) == 2
        assert chunks[0].delta == "Async t"
        assert chunks[1].delta == "est"
        assert chunks[1].message.content == "Async test"

    @pytest.mark.asyncio
    async def test_astream_chat_handles_no_user_message(self) -> None:
        """Test async streaming with no user messages."""
        mock_llm = MockLLMEchoStream()

        messages = [ChatMessage(role=MessageRole.ASSISTANT, content="Only assistant")]
        stream = await mock_llm.astream_chat(messages)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].delta == ""
        assert chunks[0].message.content == ""

    def test_metadata_properties(self) -> None:
        """Test that metadata returns expected properties."""
        mock_llm = MockLLMEchoStream()
        metadata = mock_llm.metadata

        assert metadata.context_window == 4096
        assert metadata.num_output == 256
        assert metadata.is_chat_model is True
        assert metadata.is_function_calling_model is False
        assert metadata.model_name == "mock-llm-echo"
