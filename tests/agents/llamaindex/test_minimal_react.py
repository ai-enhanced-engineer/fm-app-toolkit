"""Tests for MinimalReActAgent using TrajectoryMockLLMLlamaIndex."""

import pytest

from src.agents.llamaindex.minimal_react import MinimalReActAgent, Tool
from src.agents.llamaindex.sample_tools import calculate, get_current_time, get_weather
from src.mocks.llamaindex.mock_trajectory import TrajectoryMockLLMLlamaIndex


@pytest.mark.asyncio
async def test__single_tool_call__returns_response_with_one_source() -> None:
    """Test agent with single tool call followed by answer."""
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            'Thought: I need to get the weather for Tokyo.\nAction: get_weather\nAction Input: {"location": "Tokyo"}',
            "Thought: I have the weather information.\nAnswer: Weather in Tokyo: 75°F and sunny",
        ]
    )

    tools = [
        Tool(name="get_weather", description="Get weather for a city", function=get_weather),
    ]

    agent = MinimalReActAgent(llm=mock_llm, tools=tools, max_steps=5)
    result = await agent.run("What's the weather in Tokyo?")

    # Verify agent processed tool output correctly
    assert "Weather in Tokyo" in result["response"]
    assert len(result["reasoning"]) == 2  # Two reasoning steps (thought + answer)
    assert len(result["sources"]) == 1  # Tool was called once


@pytest.mark.asyncio
async def test__multi_step_reasoning__accumulates_multiple_sources() -> None:
    """Test agent with multiple tool calls in sequence."""
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            'Thought: I need to calculate 15 times 7 first.\nAction: calculate\nAction Input: {"expression": "15 * 7"}',
            'Thought: I got 105. Now I need to add 23.\nAction: calculate\nAction Input: {"expression": "105 + 23"}',
            "Thought: I have the final result.\nAnswer: The result of 15 times 7 plus 23 is 128.",
        ]
    )

    tools = [
        Tool(name="calculate", description="Perform math calculations", function=calculate),
    ]

    agent = MinimalReActAgent(llm=mock_llm, tools=tools, max_steps=10)
    result = await agent.run("What is 15 times 7 plus 23?")

    # Verify agent accumulated results from multiple tool calls
    assert "128" in result["response"]
    assert len(result["reasoning"]) == 3  # Three reasoning steps
    assert len(result["sources"]) == 2  # Two tool calls


@pytest.mark.asyncio
async def test__direct_answer__returns_response_with_no_sources() -> None:
    """Test agent answering directly without using tools."""
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=["Thought: This is a greeting that doesn't require tools.\nAnswer: Hello! How can I help you today?"]
    )

    tools = [
        Tool(name="get_weather", description="Get weather for a city", function=get_weather),
        Tool(name="calculate", description="Perform math calculations", function=calculate),
    ]

    agent = MinimalReActAgent(llm=mock_llm, tools=tools, max_steps=5)
    result = await agent.run("Hello!")

    # Verify agent answered directly without tools
    assert "Hello" in result["response"]
    assert len(result["reasoning"]) == 1  # Single reasoning step
    assert len(result["sources"]) == 0  # No tools used


@pytest.mark.asyncio
async def test__max_steps_reached__returns_incomplete_message() -> None:
    """Test agent reaching max steps limit."""
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            'Thought: Let me check the weather.\nAction: get_weather\nAction Input: {"location": "Tokyo"}',
            'Thought: Let me check again.\nAction: get_weather\nAction Input: {"location": "Tokyo"}',
            'Thought: Still checking.\nAction: get_weather\nAction Input: {"location": "Tokyo"}',
        ]
    )

    tools = [
        Tool(name="get_weather", description="Get weather for a city", function=get_weather),
    ]

    agent = MinimalReActAgent(llm=mock_llm, tools=tools, max_steps=2)
    result = await agent.run("What's the weather?")

    # Verify agent stopped at max steps with appropriate message
    assert "couldn't complete" in result["response"]
    assert len(result["reasoning"]) == 2  # Hit max steps


@pytest.mark.asyncio
async def test__tool_not_found__adds_error_to_sources() -> None:
    """Test agent handling nonexistent tool gracefully."""
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            'Thought: I\'ll use a nonexistent tool.\nAction: magic_tool\nAction Input: {"param": "value"}',
            "Thought: That didn't work. Let me provide an answer.\nAnswer: I encountered an error with the tool.",
        ]
    )

    tools = [
        Tool(name="get_weather", description="Get weather for a city", function=get_weather),
    ]

    agent = MinimalReActAgent(llm=mock_llm, tools=tools, max_steps=5)
    result = await agent.run("Test query")

    # Verify agent recovered from tool error
    assert "error" in result["response"].lower()
    assert len(result["sources"]) == 1  # Error message captured as source


@pytest.mark.asyncio
async def test__multiple_tools_available__selects_correct_tool() -> None:
    """Test agent with multiple tools, using the appropriate one."""
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            "Thought: I need the current time.\nAction: get_current_time\nAction Input: {}",
            "Thought: I have the time.\nAnswer: The current time is shown above.",
        ]
    )

    tools = [
        Tool(name="get_weather", description="Get weather for a city", function=get_weather),
        Tool(name="calculate", description="Perform math calculations", function=calculate),
        Tool(name="get_current_time", description="Get current date and time", function=get_current_time),
    ]

    agent = MinimalReActAgent(llm=mock_llm, tools=tools, max_steps=5)
    result = await agent.run("What time is it?")

    # Verify agent used the correct tool
    assert "current time" in result["response"].lower()
    assert len(result["sources"]) == 1
    assert "UTC" in result["sources"][0]  # get_current_time returns UTC time


@pytest.mark.asyncio
async def test__empty_llm_response__handles_gracefully() -> None:
    """Test agent handling empty LLM response."""
    mock_llm = TrajectoryMockLLMLlamaIndex(chain=[""])

    tools = [
        Tool(name="get_weather", description="Get weather for a city", function=get_weather),
    ]

    agent = MinimalReActAgent(llm=mock_llm, tools=tools, max_steps=2)
    result = await agent.run("Test query")

    # Verify agent handles empty response without crashing
    assert result is not None
    assert "response" in result
    assert isinstance(result["reasoning"], list)
    assert isinstance(result["sources"], list)


@pytest.mark.asyncio
async def test__malformed_action_input__continues_execution() -> None:
    """Test agent handling malformed JSON in action input."""
    mock_llm = TrajectoryMockLLMLlamaIndex(
        chain=[
            "Thought: Let me try something.\nAction: get_weather\nAction Input: {invalid json}",
            "Thought: That didn't parse. Let me answer.\nAnswer: I had trouble with the input.",
        ]
    )

    tools = [
        Tool(name="get_weather", description="Get weather for a city", function=get_weather),
    ]

    agent = MinimalReActAgent(llm=mock_llm, tools=tools, max_steps=5)
    result = await agent.run("Test query")

    # Verify agent recovers from malformed input
    assert result is not None
    assert "response" in result
