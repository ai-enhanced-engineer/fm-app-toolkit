"""Tests for mock LLM implementations demonstrating deterministic testing patterns."""

import pytest
from llama_cloud import MessageRole
from llama_index.core.base.llms.types import ChatMessage

# ----------------------------------------------
# MockLLMWithChain TESTS - Sequential Response Control
# ----------------------------------------------


def test_mock_llm_chain_sequential_responses(mock_llm_factory):
    """Demonstrate controlling exact LLM response sequences for deterministic testing.
    
    MockLLMWithChain enables testing complex agent workflows by providing
    predetermined responses in sequence, ensuring reproducible test scenarios.
    """
    chain = ["First response", "Second response", "Third response"]
    mock_llm = mock_llm_factory(chain)
    
    # Each call returns the next response in sequence
    response1 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Q1")])
    assert response1.message.content == "First response"
    
    response2 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Q2")])
    assert response2.message.content == "Second response"
    
    response3 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Q3")])
    assert response3.message.content == "Third response"
    
    # Beyond chain length returns empty response
    response4 = mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Q4")])
    assert response4.message.content == ""


def test_mock_llm_chain_streaming_preserves_content(mock_llm_factory):
    """Demonstrate that streaming maintains exact message formatting.
    
    Critical for testing ReAct agents where precise formatting of
    Thought/Action/Action Input matters for parsing.
    """
    # ReAct-style response with specific formatting
    chain = [
        "Thought: I need to search for information.\nAction: search\nAction Input: {'query': 'test data'}",
        "Thought: Found results.\nAnswer: Here are the search results.",
    ]
    mock_llm = mock_llm_factory(chain)
    
    # Stream first message and verify exact content preservation
    stream = mock_llm.stream_chat([ChatMessage(role=MessageRole.USER, content="Query")])
    chunks = list(stream)
    full_content = "".join(c.delta for c in chunks)
    assert full_content == chain[0]
    
    # Verify character-by-character streaming
    assert len(chunks) == len(chain[0])
    assert chunks[0].delta == "T"
    assert chunks[-1].message.content == chain[0]  # Final chunk has full content


def test_mock_llm_chain_reset_for_test_isolation(mock_llm_factory):
    """Demonstrate reset capability for test isolation.
    
    Reset allows reusing the same mock instance across multiple
    test scenarios without state pollution.
    """
    chain = ["Response A", "Response B"]
    mock_llm = mock_llm_factory(chain)
    
    # First run through the chain
    assert mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Q1")]).message.content == "Response A"
    assert mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Q2")]).message.content == "Response B"
    
    # Reset and replay from beginning
    mock_llm.reset()
    assert mock_llm.chat([ChatMessage(role=MessageRole.USER, content="Q3")]).message.content == "Response A"


@pytest.mark.asyncio
async def test_mock_llm_chain_async_streaming(mock_llm_factory):
    """Demonstrate async streaming for testing async agent workflows."""
    chain = ["Async response"]
    mock_llm = mock_llm_factory(chain)
    
    stream = await mock_llm.astream_chat([ChatMessage(role=MessageRole.USER, content="Query")])
    
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    
    # Verify async streaming maintains content integrity
    full_content = "".join(c.delta for c in chunks)
    assert full_content == "Async response"
    assert len(chunks) == len("Async response")


# ----------------------------------------------
# MockLLMEchoStream TESTS - Input Echo Behavior
# ----------------------------------------------


def test_mock_llm_echo_basic_echo(mock_llm_echo):
    """Demonstrate echo behavior for testing message flow.
    
    MockLLMEchoStream is useful for testing components that process
    LLM outputs without needing specific response content.
    """
    messages = [
        ChatMessage(role=MessageRole.ASSISTANT, content="Previous response"),
        ChatMessage(role=MessageRole.USER, content="Echo this message"),
    ]
    
    response = mock_llm_echo.chat(messages)
    assert response.message.content == "Echo this message"
    
    # Empty messages return empty response
    assert mock_llm_echo.chat([]).message.content == ""


def test_mock_llm_echo_streaming_chunks(mock_llm_echo):
    """Demonstrate chunked streaming for testing streaming handlers.
    
    Messages are split into 7-character chunks to test streaming
    buffer management and progressive content building.
    """
    # Test exact chunk size (7 chars)
    messages = [ChatMessage(role=MessageRole.USER, content="1234567")]
    stream = mock_llm_echo.stream_chat(messages)
    chunks = list(stream)
    
    assert len(chunks) == 1
    assert chunks[0].delta == "1234567"
    
    # Test multiple chunks (10 chars = 7 + 3)
    messages = [ChatMessage(role=MessageRole.USER, content="1234567890")]
    stream = mock_llm_echo.stream_chat(messages)
    chunks = list(stream)
    
    assert len(chunks) == 2
    assert chunks[0].delta == "1234567"
    assert chunks[1].delta == "890"
    assert chunks[1].message.content == "1234567890"  # Final chunk has full content


@pytest.mark.parametrize("content,expected_chunks", [
    ("", []),  # Empty content
    ("Hi", ["Hi"]),  # Less than chunk size
    ("1234567", ["1234567"]),  # Exactly chunk size
    ("A" * 20, ["A" * 7, "A" * 7, "A" * 6]),  # Multiple chunks
])
def test_mock_llm_echo_various_message_lengths(mock_llm_echo, content, expected_chunks):
    """Test streaming behavior with various message lengths.
    
    Validates edge cases in streaming: empty, partial, exact, and multiple chunks.
    """
    messages = [ChatMessage(role=MessageRole.USER, content=content)]
    stream = mock_llm_echo.stream_chat(messages)
    
    chunks = [chunk.delta for chunk in stream]
    assert chunks == expected_chunks
    
    # Verify reassembly if content exists
    if chunks:
        stream = mock_llm_echo.stream_chat(messages)
        *_, last_chunk = stream
        assert last_chunk.message.content == content


@pytest.mark.asyncio
async def test_mock_llm_echo_async_cumulative_building(mock_llm_echo):
    """Test async streaming builds content progressively.
    
    Each chunk contains the cumulative content built so far,
    essential for UI components showing progressive responses.
    """
    test_content = "Progressive build test"  # 22 chars = 4 chunks
    messages = [ChatMessage(role=MessageRole.USER, content=test_content)]
    
    stream = await mock_llm_echo.astream_chat(messages)
    cumulative_content = ""
    
    async for chunk in stream:
        cumulative_content += chunk.delta
        assert chunk.message.content == cumulative_content
        assert chunk.message.role == MessageRole.ASSISTANT
    
    assert cumulative_content == test_content


# ----------------------------------------------
# METADATA TESTS - Model Properties
# ----------------------------------------------


def test_mock_llm_metadata_properties(mock_llm_factory, mock_llm_echo):
    """Verify mock LLMs provide expected metadata for compatibility."""
    # MockLLMWithChain metadata
    chain_llm = mock_llm_factory(["test"])
    chain_metadata = chain_llm.metadata
    assert chain_metadata.context_window == 4096
    assert chain_metadata.is_chat_model is True
    assert chain_metadata.model_name == "mock-llm-chain"
    
    # MockLLMEchoStream metadata
    echo_metadata = mock_llm_echo.metadata
    assert echo_metadata.context_window == 4096
    assert echo_metadata.is_chat_model is True
    assert echo_metadata.model_name == "mock-llm-echo"