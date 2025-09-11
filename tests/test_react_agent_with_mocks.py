"""Tests demonstrating ReActAgent with mocked LLMs for deterministic testing."""

import pytest
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool

from fm_app_toolkit.testing.mocks import MockLLMEchoStream, MockLLMWithChain
from fm_app_toolkit.tools import add, divide, multiply, reverse_string, word_count


@pytest.mark.asyncio
async def test__react_agent__predefined_responses() -> None:
    """Test that ReActAgent executes correctly with predefined mock LLM responses."""
    # Create a mock LLM with predefined ReAct-formatted responses
    mock_llm = MockLLMWithChain(
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
    mock_llm = MockLLMWithChain(
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
    mock_llm = MockLLMWithChain(
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
    mock_llm = MockLLMWithChain(
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
    mock_llm = MockLLMWithChain(
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
    mock_llm = MockLLMWithChain(
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
    mock_llm = MockLLMWithChain(
        chain=[
            "Thought: I'll provide information about Python.\nAnswer: Python is a high-level programming language."
        ]
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
    mock_llm = MockLLMWithChain(
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