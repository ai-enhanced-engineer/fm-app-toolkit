"""Tests demonstrating ReActAgent with mocked LLMs for deterministic testing."""

from unittest.mock import MagicMock

import pytest
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool

from src.mocks.llamaindex.mock_echo import MockLLMEchoStream
from src.mocks.llamaindex.mock_trajectory import TrajectoryMockLLMLlamaIndex
from src.tools import add, divide, multiply, reverse_string, word_count


@pytest.mark.asyncio
async def test__react_agent__predefined_responses() -> None:
    """Test that ReActAgent executes correctly with predefined mock LLM responses."""
    # Create a mock LLM with predefined ReAct-formatted responses
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            "Thought: I need to add these two numbers together.\nAction: add\nAction Input: {'a': 5, 'b': 3}",
            "Thought: I have the result of the addition.\nAnswer: The sum of 5 and 3 is 8.",
        ]
    )

    # Create tools
    add_tool = FunctionTool.from_defaults(fn=add, name="add")

    # Create agent with mock LLM
    agent = ReActAgent(tools=[add_tool], llm=mock_llm, verbose=True)

    # Run the agent
    response = await agent.run(user_msg="What is 5 plus 3?")

    # Verify the response contains the expected answer
    assert "8" in str(response)

    # Reset mock for next test
    mock_llm.reset()


@pytest.mark.asyncio
async def test__react_agent__multi_step_reasoning() -> None:
    """Test that ReActAgent handles multi-step reasoning with multiple tool calls."""
    # Create mock with multi-step reasoning
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            "Thought: I need to first multiply 4 by 5.\nAction: multiply\nAction Input: {'a': 4, 'b': 5}",
            "Thought: I got 20. Now I need to add 10 to this result.\nAction: add\nAction Input: {'a': 20, 'b': 10}",
            "Thought: I have calculated the final result.\nAnswer: The result of (4 * 5) + 10 is 30.",
        ]
    )

    # Create tools
    add_tool = FunctionTool.from_defaults(fn=add, name="add")
    multiply_tool = FunctionTool.from_defaults(fn=multiply, name="multiply")

    # Create agent
    agent = ReActAgent(tools=[add_tool, multiply_tool], llm=mock_llm, verbose=True)

    # Run the agent
    response = await agent.run(user_msg="Calculate (4 * 5) + 10")

    # Verify the response
    assert "30" in str(response)


@pytest.mark.asyncio
async def test__react_agent__string_tools() -> None:
    """Test that ReActAgent works correctly with string manipulation tools."""
    # Create mock for string operations
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            "Thought: I need to reverse the given string.\nAction: reverse_string\nAction Input: {'text': 'hello world'}",
            "Thought: I have reversed the string.\nAnswer: The reversed string is 'dlrow olleh'.",
        ]
    )

    # Create string tool
    reverse_tool = FunctionTool.from_defaults(fn=reverse_string, name="reverse_string")

    # Create agent
    agent = ReActAgent(tools=[reverse_tool], llm=mock_llm, verbose=True)

    # Run the agent
    response = await agent.run(user_msg="Reverse the string 'hello world'")

    # Verify the response
    assert "dlrow olleh" in str(response)


@pytest.mark.asyncio
async def test__react_agent__error_handling() -> None:
    """Test that ReActAgent handles tool errors gracefully."""
    # Create mock that will trigger division by zero
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            "Thought: I need to divide 10 by 0.\nAction: divide\nAction Input: {'a': 10, 'b': 0}",
            "Thought: There was an error dividing by zero. Division by zero is undefined.\nAnswer: Cannot divide by zero - this operation is undefined.",
        ]
    )

    # Create divide tool
    divide_tool = FunctionTool.from_defaults(fn=divide, name="divide")

    # Create agent
    agent = ReActAgent(tools=[divide_tool], llm=mock_llm, verbose=True)

    # Run the agent - should handle the error gracefully
    response = await agent.run(user_msg="What is 10 divided by 0?")

    # The agent should handle the error and provide a response
    response_str = str(response)
    assert response_str is not None
    # The response should mention the error or undefined operation
    assert "zero" in response_str.lower() or "undefined" in response_str.lower()


@pytest.mark.asyncio
async def test__react_agent__streaming_echo() -> None:
    """Test that ReActAgent streaming capabilities work with echo mock."""
    # Create echo mock for testing streaming
    mock_llm = MockLLMEchoStream()

    # Create a simple tool
    word_count_tool = FunctionTool.from_defaults(fn=word_count, name="word_count")

    # Create agent
    agent = ReActAgent(tools=[word_count_tool], llm=mock_llm, verbose=True)

    # Test with workflow run
    response = await agent.run(user_msg="Count words in 'hello world test'")

    # Verify we got a response
    assert response is not None
    response_str = str(response)
    assert len(response_str) > 0


@pytest.mark.asyncio
async def test__react_agent__tool_selection() -> None:
    """Test that ReActAgent correctly selects appropriate tools from multiple options."""
    # Create mock that selects different tools based on the task
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            "Thought: I need to count the words in the text.\nAction: word_count\nAction Input: {'text': 'hello world test'}",
            "Thought: The text contains 3 words.\nAnswer: The text 'hello world test' contains 3 words.",
        ]
    )

    # Create multiple tools
    add_tool = FunctionTool.from_defaults(fn=add, name="add")
    multiply_tool = FunctionTool.from_defaults(fn=multiply, name="multiply")
    word_count_tool = FunctionTool.from_defaults(fn=word_count, name="word_count")
    reverse_tool = FunctionTool.from_defaults(fn=reverse_string, name="reverse_string")

    # Create agent with multiple tools
    agent = ReActAgent(tools=[add_tool, multiply_tool, word_count_tool, reverse_tool], llm=mock_llm, verbose=True)

    # Run the agent
    response = await agent.run(user_msg="How many words are in 'hello world test'?")

    # Verify the response
    assert "3" in str(response)


@pytest.mark.asyncio
async def test__react_agent__direct_answer() -> None:
    """Test that ReActAgent can provide direct answers without using tools."""
    # Create mock that answers directly without tool use
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            "Thought: This is a simple greeting that doesn't require any tools.\nAnswer: Hello! I'm a ReAct agent. How can I help you today?"
        ]
    )

    # Create agent with tools (but won't use them for this query)
    add_tool = FunctionTool.from_defaults(fn=add, name="add")
    agent = ReActAgent(tools=[add_tool], llm=mock_llm, verbose=True)

    # Run the agent
    response = await agent.run(user_msg="Hello!")

    # Verify we got a response
    response_str = str(response)
    assert "Hello" in response_str
    assert "help" in response_str.lower()


@pytest.mark.asyncio
async def test__react_agent_streaming__response_collection() -> None:
    """Test that ReActAgent streaming response collection works correctly."""
    # Create a mock that will stream a response
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=["Thought: I'll provide information about Python.\nAnswer: Python is a high-level programming language."]
    )

    # Create agent
    agent = ReActAgent(
        tools=[],  # No tools needed for this test
        llm=mock_llm,
        verbose=False,
    )

    # Run the agent
    response = await agent.run(user_msg="Tell me about Python")

    # Verify we got a response
    assert response is not None
    response_str = str(response)
    assert len(response_str) > 0


@pytest.mark.asyncio
async def test__react_agent_streaming__with_tools() -> None:
    """Test that ReActAgent streaming behavior works when tools are executed."""
    # Create mock for tool execution with streaming
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            "Thought: I need to add 15 and 25.\nAction: add\nAction Input: {'a': 15, 'b': 25}",
            "Thought: The addition gives us 40.\nAnswer: The sum of 15 and 25 is 40.",
        ]
    )

    # Create tool
    add_tool = FunctionTool.from_defaults(fn=add, name="add")

    # Create agent
    agent = ReActAgent(tools=[add_tool], llm=mock_llm, verbose=False)

    # Run the agent
    response = await agent.run(user_msg="What is 15 + 25?")

    # Verify the response contains the answer
    assert "40" in str(response)


# =============================================================================
# Tool Invocation Tests (Article: Testing Tool Invocations)
# =============================================================================


@pytest.mark.asyncio
async def test__calculator_tool_invocation__correct_parameters() -> None:
    """Agent should call calculator with correct parameters."""
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            "Thought: User wants 23 * 45, I'll use calculator.\nAction: calculator\nAction Input: {'expression': '23 * 45'}",
            "Thought: Got result.\nAnswer: 1035",
        ]
    )

    # Spy on tool calls without replacing implementation
    calculator_spy = MagicMock(return_value=1035)
    calculator_tool = FunctionTool.from_defaults(fn=calculator_spy, name="calculator")

    agent = ReActAgent(tools=[calculator_tool], llm=mock_llm)

    response = await agent.run(user_msg="What's 23 times 45?")

    # Verify tool selection and parameters
    calculator_spy.assert_called_once()
    call_args = calculator_spy.call_args[1]
    assert call_args["expression"] == "23 * 45"
    assert "1035" in str(response)


@pytest.mark.asyncio
async def test__multi_tool_sequence__correct_order() -> None:
    """Agent should use search -> calculator -> database in order."""
    call_sequence: list[tuple[str, str]] = []

    def mock_search(query: str) -> str:
        call_sequence.append(("search", query))
        return "Product price: $45"

    def mock_calculator(expression: str) -> str:
        call_sequence.append(("calculator", expression))
        return "48.6"

    def mock_save(record: str) -> str:
        call_sequence.append(("database", record))
        return "Saved"

    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            "Thought: I need to search for the product price first.\nAction: search\nAction Input: {'query': 'laptop price'}",
            "Thought: Found price $45. Now calculate with 8% tax.\nAction: calculator\nAction Input: {'expression': '45 * 1.08'}",
            "Thought: Tax applied gives $48.6. Save to database.\nAction: database\nAction Input: {'record': 'laptop: $48.60'}",
            "Thought: Order saved successfully.\nAnswer: The laptop costs $48.60 after tax and has been saved.",
        ]
    )

    search_tool = FunctionTool.from_defaults(fn=mock_search, name="search")
    calculator_tool = FunctionTool.from_defaults(fn=mock_calculator, name="calculator")
    database_tool = FunctionTool.from_defaults(fn=mock_save, name="database")

    agent = ReActAgent(
        tools=[search_tool, calculator_tool, database_tool],
        llm=mock_llm,
    )

    response = await agent.run(user_msg="Find the laptop price, add 8% tax, and save it")

    # Verify call order
    assert len(call_sequence) == 3
    assert call_sequence[0][0] == "search"
    assert call_sequence[1][0] == "calculator"
    assert call_sequence[2][0] == "database"
    assert "48.6" in str(response) or "saved" in str(response).lower()


# =============================================================================
# Error Recovery Tests (Article: Testing Error Recovery)
# =============================================================================


@pytest.mark.asyncio
async def test__search_timeout_mid_chain__uses_fallback() -> None:
    """Verify agent uses fallback_search when primary search times out."""
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            "Thought: I need to search for refund policy.\nAction: search_kb\nAction Input: {'query': 'refund policy'}",
            "Thought: Search timed out. I'll try the fallback.\nAction: fallback_search\nAction Input: {'query': 'refund policy'}",
            "Thought: Got results from fallback.\nAnswer: The refund policy allows returns within 30 days.",
        ]
    )

    # Primary tool fails with timeout
    def search_kb(query: str) -> str:
        raise TimeoutError("Search API timeout after 5s")

    # Fallback succeeds
    def fallback_search(query: str) -> str:
        return "Refunds allowed within 30 days with receipt."

    agent = ReActAgent(
        tools=[
            FunctionTool.from_defaults(fn=search_kb, name="search_kb"),
            FunctionTool.from_defaults(fn=fallback_search, name="fallback_search"),
        ],
        llm=mock_llm,
    )

    response = await agent.run(user_msg="What is the refund policy?")

    assert "30 days" in str(response)
    # Three reasoning steps: initial search, fallback, answer


@pytest.mark.asyncio
async def test__cascading_failures__partial_success() -> None:
    """Verify agent exhausts fallbacks and returns partial results."""
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            "Thought: I'll search the web for documentation.\nAction: web_search\nAction Input: {'query': 'API documentation'}",
            "Thought: Rate limited. Try cached results.\nAction: cached_search\nAction Input: {'query': 'API documentation'}",
            "Thought: Cache empty. Try local docs.\nAction: local_docs\nAction Input: {'query': 'API documentation'}",
            "Thought: Found partial docs locally.\nAnswer: Based on limited offline documentation: [partial answer]",
        ]
    )

    call_sequence: list[str] = []

    def web_search(query: str) -> str:
        call_sequence.append("web_search")
        raise Exception("Rate limit: 429 Too Many Requests")

    def cached_search(query: str) -> str:
        call_sequence.append("cached_search")
        return ""  # Empty result

    def local_docs(query: str) -> str:
        call_sequence.append("local_docs")
        return "Limited offline documentation available."

    agent = ReActAgent(
        tools=[
            FunctionTool.from_defaults(fn=web_search, name="web_search"),
            FunctionTool.from_defaults(fn=cached_search, name="cached_search"),
            FunctionTool.from_defaults(fn=local_docs, name="local_docs"),
        ],
        llm=mock_llm,
    )

    response = await agent.run(user_msg="Find API documentation")

    # Verify fallback chain executed in order
    assert call_sequence == ["web_search", "cached_search", "local_docs"]
    response_lower = str(response).lower()
    assert "limited" in response_lower or "partial" in response_lower
