"""Tests for mock LLM implementations."""

from typing import Callable

import pytest
from llama_cloud import MessageRole
from llama_index.core.base.llms.types import ChatMessage

from fm_app_toolkit.testing.mocks import MockLLMEchoStream, MockLLMWithChain

# ----------------------------------------------
# FIXTURES
# ----------------------------------------------


@pytest.fixture
def mock_llm_with_chain() -> Callable[[list[str]], MockLLMWithChain]:
    """Fixture to create MockLLMWithChain instances."""

    def _create(chain: list[str]) -> MockLLMWithChain:
        return MockLLMWithChain(chain=chain)

    return _create


@pytest.fixture
def mock_llm_echo() -> MockLLMEchoStream:
    """Fixture for MockLLMEchoStream instance."""
    return MockLLMEchoStream()


@pytest.fixture
def sample_messages() -> list[ChatMessage]:
    """Common test messages for echo testing."""
    return [
        ChatMessage(role=MessageRole.USER, content="Hello, world!"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Previous response"),
        ChatMessage(role=MessageRole.USER, content="Echo this"),
    ]


# ----------------------------------------------
# MockLLMWithChain TESTS
# ----------------------------------------------


def test_mock_llm_chain_init_creates_message_chain(mock_llm_with_chain: Callable) -> None:
    """Test that initialization creates the correct message chain."""
    chain = ["First response", "Second response", "Third response"]
    mock_llm = mock_llm_with_chain(chain)

    assert len(mock_llm.message_chain) == 3
    assert all(msg.role == MessageRole.ASSISTANT for msg in mock_llm.message_chain)
    assert mock_llm.message_chain[0].content == "First response"
    assert mock_llm.message_chain[1].content == "Second response"
    assert mock_llm.message_chain[2].content == "Third response"


def test_mock_llm_chain_returns_messages_in_sequence(mock_llm_with_chain: Callable) -> None:
    """Test that chat returns messages in the correct sequence."""
    chain = ["Response 1", "Response 2", "Response 3"]
    mock_llm = mock_llm_with_chain(chain)

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


def test_mock_llm_chain_stream_chat_streams_characters(mock_llm_with_chain: Callable) -> None:
    """Test that stream_chat streams response character by character."""
    chain = ["Hello"]
    mock_llm = mock_llm_with_chain(chain)

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
async def test_mock_llm_chain_astream_chat_async_streaming(mock_llm_with_chain: Callable) -> None:
    """Test async streaming of responses."""
    chain = ["Test"]
    mock_llm = mock_llm_with_chain(chain)

    messages = [ChatMessage(role=MessageRole.USER, content="Query")]
    stream = await mock_llm.astream_chat(messages)

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == 4  # "Test" has 4 characters
    assert "".join(c.delta for c in chunks) == "Test"


def test_mock_llm_chain_reset_replays_chain(mock_llm_with_chain: Callable) -> None:
    """Test that reset allows replaying the chain."""
    chain = ["First", "Second"]
    mock_llm = mock_llm_with_chain(chain)

    # Use up the chain
    response1 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Q1")])
    response2 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Q2")])
    assert response1.message.content == "First"
    assert response2.message.content == "Second"

    # Reset and replay
    mock_llm.reset()
    response3 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Q3")])
    assert response3.message.content == "First"


def test_mock_llm_chain_metadata_properties(mock_llm_with_chain: Callable) -> None:
    """Test that metadata returns expected properties."""
    mock_llm = mock_llm_with_chain(["test"])
    metadata = mock_llm.metadata

    assert metadata.context_window == 4096
    assert metadata.num_output == 256
    assert metadata.is_chat_model is True
    assert metadata.is_function_calling_model is False
    assert metadata.model_name == "mock-llm-chain"


def test_mock_llm_chain_stream_preserves_full_messages(mock_llm_with_chain: Callable) -> None:
    """Test that streaming preserves the full message content from chain.

    This test validates that character-by-character streaming doesn't corrupt
    the original messages, which is especially important for ReAct agent testing
    where exact formatting matters.
    """
    chain = [
        "Thought: I need to search for information.\nAction: search\nAction Input: {'query': 'test data'}",
        "Thought: Found results.\nAnswer: Here are the search results.",
    ]
    mock_llm = mock_llm_with_chain(chain)

    # Stream first message and verify full content
    stream1 = mock_llm.stream_chat([ChatMessage(role=MessageRole.USER, content="Query 1")])
    chunks1 = list(stream1)
    full_content1 = "".join(c.delta for c in chunks1)
    assert full_content1 == chain[0]

    # Stream second message and verify full content
    stream2 = mock_llm.stream_chat([ChatMessage(role=MessageRole.USER, content="Query 2")])
    chunks2 = list(stream2)
    full_content2 = "".join(c.delta for c in chunks2)
    assert full_content2 == chain[1]


@pytest.mark.asyncio
async def test_mock_llm_chain_astream_preserves_full_messages(mock_llm_with_chain: Callable) -> None:
    """Test that async streaming preserves the full message content from chain."""
    chain = [
        "Thought: Processing request.\nAction: calculate\nAction Input: {'value': 42}",
        "Thought: Calculation complete.\nAnswer: The result is 42.",
    ]
    mock_llm = mock_llm_with_chain(chain)

    # Async stream first message
    stream1 = await mock_llm.astream_chat([ChatMessage(role=MessageRole.USER, content="Q1")])
    chunks1 = []
    async for chunk in stream1:
        chunks1.append(chunk)
    full_content1 = "".join(c.delta for c in chunks1)
    assert full_content1 == chain[0]

    # Async stream second message
    stream2 = await mock_llm.astream_chat([ChatMessage(role=MessageRole.USER, content="Q2")])
    chunks2 = []
    async for chunk in stream2:
        chunks2.append(chunk)
    full_content2 = "".join(c.delta for c in chunks2)
    assert full_content2 == chain[1]


# ----------------------------------------------
# MockLLMEchoStream TESTS
# ----------------------------------------------


def test_mock_llm_echo_chat_echoes_user_message(
    mock_llm_echo: MockLLMEchoStream, sample_messages: list[ChatMessage]
) -> None:
    """Test that chat echoes the most recent user message."""
    response = mock_llm_echo.chat(sample_messages)
    assert response.message.content == "Echo this"


def test_mock_llm_echo_handles_empty_messages(mock_llm_echo: MockLLMEchoStream) -> None:
    """Test handling of empty message list."""
    response = mock_llm_echo.chat([])
    assert response.message.content == ""


def test_mock_llm_echo_stream_chunks_correctly(mock_llm_echo: MockLLMEchoStream) -> None:
    """Test that streaming chunks messages correctly."""
    # Test with message exactly CHUNK_SIZE
    messages = [ChatMessage(role=MessageRole.USER, content="1234567")]  # 7 chars
    stream = mock_llm_echo.stream_chat(messages)
    chunks = list(stream)

    assert len(chunks) == 1
    assert chunks[0].delta == "1234567"
    assert chunks[0].message.content == "1234567"

    # Test with longer message
    messages = [ChatMessage(role=MessageRole.USER, content="1234567890")]  # 10 chars
    mock_llm_echo = MockLLMEchoStream()  # Reset
    stream = mock_llm_echo.stream_chat(messages)
    chunks = list(stream)

    assert len(chunks) == 2
    assert chunks[0].delta == "1234567"
    assert chunks[1].delta == "890"
    assert chunks[1].message.content == "1234567890"


@pytest.mark.asyncio
async def test_mock_llm_echo_astream_chat_async(mock_llm_echo: MockLLMEchoStream) -> None:
    """Test async streaming echo."""
    messages = [ChatMessage(role=MessageRole.USER, content="Async test")]
    stream = await mock_llm_echo.astream_chat(messages)

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    # "Async test" = 10 chars, should be 2 chunks (7 + 3)
    assert len(chunks) == 2
    assert chunks[0].delta == "Async t"
    assert chunks[1].delta == "est"
    assert chunks[1].message.content == "Async test"


@pytest.mark.asyncio
async def test_mock_llm_echo_astream_handles_no_user_message(mock_llm_echo: MockLLMEchoStream) -> None:
    """Test async streaming with no user messages."""
    messages = [ChatMessage(role=MessageRole.ASSISTANT, content="Only assistant")]
    stream = await mock_llm_echo.astream_chat(messages)

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].delta == ""
    assert chunks[0].message.content == ""


def test_mock_llm_echo_metadata_properties(mock_llm_echo: MockLLMEchoStream) -> None:
    """Test that metadata returns expected properties."""
    metadata = mock_llm_echo.metadata

    assert metadata.context_window == 4096
    assert metadata.num_output == 256
    assert metadata.is_chat_model is True
    assert metadata.is_function_calling_model is False
    assert metadata.model_name == "mock-llm-echo"


# ----------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------


def generate_expected_chunks(content: str, chunk_size: int = 7) -> list[str]:
    """Generate expected chunks from content for streaming validation."""
    if not content:
        return []  # Empty content produces no chunks
    return [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]


# ----------------------------------------------
# ENHANCED MockLLMEchoStream TESTS
# ----------------------------------------------


def test_mock_llm_echo_stream_cumulative_content_building(mock_llm_echo: MockLLMEchoStream) -> None:
    """Test that streaming builds cumulative content correctly at each chunk."""
    test_content = "Tell me about the universe."  # 27 chars = 4 chunks (7+7+7+6)
    messages = [ChatMessage(role=MessageRole.USER, content=test_content)]

    # Generate expected chunks
    expected_chunks = generate_expected_chunks(test_content)
    assert len(expected_chunks) == 4  # Verify we get 4 chunks

    cumulative_content = ""
    stream = mock_llm_echo.stream_chat(messages)

    for i, response in enumerate(stream):
        # Verify delta matches expected chunk
        assert response.delta == expected_chunks[i], (
            f"Chunk {i}: expected '{expected_chunks[i]}', got '{response.delta}'"
        )

        # Build and verify cumulative content
        cumulative_content += response.delta
        assert response.message.content == cumulative_content
        assert response.message.role == MessageRole.ASSISTANT

    # Verify final content matches original
    assert cumulative_content == test_content


@pytest.mark.asyncio
async def test_mock_llm_echo_astream_cumulative_content_building(mock_llm_echo: MockLLMEchoStream) -> None:
    """Test async streaming builds cumulative content correctly at each chunk."""
    test_content = "Tell me about the universe."  # 27 chars
    messages = [ChatMessage(role=MessageRole.USER, content=test_content)]

    expected_chunks = generate_expected_chunks(test_content)
    cumulative_content = ""

    stream = await mock_llm_echo.astream_chat(messages)
    chunk_index = 0

    async for response in stream:
        # Verify delta matches expected chunk
        assert response.delta == expected_chunks[chunk_index], (
            f"Chunk {chunk_index}: expected '{expected_chunks[chunk_index]}', got '{response.delta}'"
        )

        # Build and verify cumulative content
        cumulative_content += response.delta
        assert response.message.content == cumulative_content
        assert response.message.role == MessageRole.ASSISTANT
        chunk_index += 1

    # Verify we processed all expected chunks
    assert chunk_index == len(expected_chunks)
    assert cumulative_content == test_content


@pytest.mark.parametrize(
    "content,expected_chunks",
    [
        ("", []),  # Empty content - no chunks produced
        ("Hi", ["Hi"]),  # Less than chunk size
        ("1234567", ["1234567"]),  # Exactly chunk size (7)
        ("12345678", ["1234567", "8"]),  # Slightly over
        ("A" * 14, ["A" * 7, "A" * 7]),  # Exactly 2 chunks
        ("A" * 20, ["A" * 7, "A" * 7, "A" * 6]),  # Multiple chunks with remainder
        ("Test message for chunking", ["Test me", "ssage f", "or chun", "king"]),  # Real text
    ],
)
def test_mock_llm_echo_stream_various_lengths(
    mock_llm_echo: MockLLMEchoStream, content: str, expected_chunks: list[str]
) -> None:
    """Test streaming with various message lengths."""
    messages = [ChatMessage(role=MessageRole.USER, content=content)]
    stream = mock_llm_echo.stream_chat(messages)

    chunks = [chunk.delta for chunk in stream]
    assert chunks == expected_chunks, f"Content '{content}': expected {expected_chunks}, got {chunks}"

    # Also verify the final message content is correct
    if chunks:
        stream = mock_llm_echo.stream_chat(messages)
        *_, last_chunk = stream  # Get the last chunk
        assert last_chunk.message.content == content
    elif content == "":
        # For empty content, verify we get no chunks
        assert chunks == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,expected_chunk_count",
    [
        ("", 0),  # Empty produces no chunks
        ("Short", 1),  # Less than 7 chars
        ("Exactly7", 2),  # 8 chars = 2 chunks
        ("A" * 21, 3),  # 21 chars = 3 chunks
        ("This is a longer message for testing async streaming", 8),  # 52 chars
    ],
)
async def test_mock_llm_echo_astream_various_lengths(
    mock_llm_echo: MockLLMEchoStream, content: str, expected_chunk_count: int
) -> None:
    """Test async streaming with various message lengths."""
    messages = [ChatMessage(role=MessageRole.USER, content=content)]
    stream = await mock_llm_echo.astream_chat(messages)

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == expected_chunk_count, (
        f"Content length {len(content)}: expected {expected_chunk_count} chunks, got {len(chunks)}"
    )

    # Verify reassembled content
    reassembled = "".join(chunk.delta for chunk in chunks)
    assert reassembled == content
