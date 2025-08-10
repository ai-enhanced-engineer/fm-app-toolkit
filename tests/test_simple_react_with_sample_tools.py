"""Integration tests for SimpleReActAgent with sample tools.

These tests demonstrate how the agent can use various sample tools
to answer queries through reasoning and tool execution.
"""

from typing import Callable
from unittest.mock import patch

import pytest

from ai_test_lab.agents.sample_tools import (
    calculate,
    flip_coin,
    get_current_time,
    get_joke,
    get_random_fact,
    get_weather,
    roll_dice,
    search_web,
)
from ai_test_lab.agents.simple_react import SimpleReActAgent, Tool
from ai_test_lab.testing.mocks import MockLLMWithChain


@pytest.fixture
def create_agent_with_sample_tools() -> Callable:
    """Fixture to create an agent with sample tools."""
    def _create(chain: list[str], tool_functions: list, verbose: bool = False) -> SimpleReActAgent:
        # Convert functions to Tool objects
        tools = []
        for func in tool_functions:
            # Extract function name and docstring for description
            name = func.__name__
            description = func.__doc__.split('\n')[0] if func.__doc__ else f"Tool: {name}"
            tools.append(Tool(name=name, function=func, description=description))
        
        mock_llm = MockLLMWithChain(chain=chain)
        return SimpleReActAgent(
            llm=mock_llm,
            system_header="You are a helpful assistant with access to various tools.",
            tools=tools,
            verbose=verbose
        )
    return _create


@pytest.mark.asyncio
async def test_agent_gets_current_time(create_agent_with_sample_tools: Callable) -> None:
    """Test agent using get_current_time tool."""
    agent = create_agent_with_sample_tools(
        chain=[
            "Thought: The user wants to know the current time. I'll use the get_current_time tool.\nAction: get_current_time\nAction Input: {}",
            "Thought: I have retrieved the current time successfully.\nAnswer: The current time is 2024-01-15 14:30:00 UTC"
        ],
        tool_functions=[get_current_time]
    )
    
    result = await agent.run("What time is it?")
    
    assert "current time" in result["response"].lower()
    assert len(result["reasoning"]) == 3  # Thought-Action, Observation, Thought-Answer
    assert len(result["sources"]) == 1  # One tool output


@pytest.mark.asyncio
async def test_agent_performs_calculation(create_agent_with_sample_tools: Callable) -> None:
    """Test agent using calculate tool."""
    agent = create_agent_with_sample_tools(
        chain=[
            "Thought: I need to calculate 15 * 7 for the user.\nAction: calculate\nAction Input: {\"expression\": \"15 * 7\"}",
            "Thought: The calculation shows that 15 * 7 equals 105.\nAnswer: 15 multiplied by 7 equals 105."
        ],
        tool_functions=[calculate]
    )
    
    result = await agent.run("What is 15 times 7?")
    
    assert "105" in result["response"]
    assert len(result["sources"]) == 1
    assert "15 * 7 = 105" in result["sources"][0]


@pytest.mark.asyncio
@patch('random.choice')
@patch('random.randint')
async def test_agent_checks_weather(mock_randint, mock_choice, create_agent_with_sample_tools: Callable) -> None:
    """Test agent using get_weather tool with mocked random values."""
    mock_choice.return_value = "sunny"
    mock_randint.return_value = 72
    
    agent = create_agent_with_sample_tools(
        chain=[
            "Thought: The user wants to know the weather in Paris. I'll check the weather.\nAction: get_weather\nAction Input: {\"location\": \"Paris\"}",
            "Thought: The weather information shows it's 72°F and sunny in Paris.\nAnswer: The weather in Paris is currently 72°F and sunny."
        ],
        tool_functions=[get_weather]
    )
    
    result = await agent.run("What's the weather like in Paris?")
    
    assert "72°F" in result["response"]
    assert "sunny" in result["response"]
    assert "Weather in Paris: 72°F and sunny" in result["sources"][0]


@pytest.mark.asyncio
async def test_agent_searches_web(create_agent_with_sample_tools: Callable) -> None:
    """Test agent using search_web tool."""
    agent = create_agent_with_sample_tools(
        chain=[
            "Thought: I need to search for information about quantum computing.\nAction: search_web\nAction Input: {\"query\": \"quantum computing basics\"}",
            "Thought: I found some information about quantum computing basics.\nAnswer: I found information about quantum computing basics. Here's a relevant article on quantum computing basics that explains the fundamental concepts."
        ],
        tool_functions=[search_web]
    )
    
    result = await agent.run("Search for information about quantum computing basics")
    
    assert "quantum computing" in result["response"].lower()
    assert len(result["sources"]) == 1
    assert "Found information about quantum computing basics" in result["sources"][0]


@pytest.mark.asyncio
@patch('random.randint')
async def test_agent_rolls_dice(mock_randint, create_agent_with_sample_tools: Callable) -> None:
    """Test agent using roll_dice tool."""
    mock_randint.return_value = 18
    
    agent = create_agent_with_sample_tools(
        chain=[
            "Thought: The user wants me to roll a 20-sided dice. I'll use the roll_dice tool.\nAction: roll_dice\nAction Input: {\"sides\": 20}",
            "Thought: I rolled the dice and got 18.\nAnswer: I rolled a 20-sided dice and got 18!"
        ],
        tool_functions=[roll_dice]
    )
    
    result = await agent.run("Roll a d20 for me")
    
    assert "18" in result["response"]
    assert "Rolled a 20-sided dice: 18" in result["sources"][0]


@pytest.mark.asyncio
@patch('random.choice')
async def test_agent_flips_coin(mock_choice, create_agent_with_sample_tools: Callable) -> None:
    """Test agent using flip_coin tool."""
    mock_choice.return_value = "Heads"
    
    agent = create_agent_with_sample_tools(
        chain=[
            "Thought: The user wants me to flip a coin. I'll use the flip_coin tool.\nAction: flip_coin\nAction Input: {}",
            "Thought: The coin landed on Heads.\nAnswer: I flipped the coin and it landed on Heads!"
        ],
        tool_functions=[flip_coin]
    )
    
    result = await agent.run("Flip a coin")
    
    assert "Heads" in result["response"]
    assert "Coin flip: Heads" in result["sources"][0]


@pytest.mark.asyncio
async def test_agent_multi_tool_reasoning(create_agent_with_sample_tools: Callable) -> None:
    """Test agent using multiple tools in sequence."""
    with patch('random.randint') as mock_randint, patch('random.choice') as mock_choice:
        mock_randint.side_effect = [6, 4]  # First for dice, second for dice
        mock_choice.return_value = "Tails"  # For coin flip
        
        agent = create_agent_with_sample_tools(
            chain=[
                "Thought: I need to roll two dice first.\nAction: roll_dice\nAction Input: {}",
                "Thought: First dice shows 6. Let me roll another.\nAction: roll_dice\nAction Input: {}",
                "Thought: Second dice shows 4. Now let me flip a coin.\nAction: flip_coin\nAction Input: {}",
                "Thought: The coin shows Tails. So the results are: two dice showing 6 and 4 (total 10), and a coin showing Tails.\nAnswer: I rolled two dice and got 6 and 4 (total of 10), and flipped a coin which landed on Tails."
            ],
            tool_functions=[roll_dice, flip_coin]
        )
        
        result = await agent.run("Roll two dice and flip a coin")
        
        assert "6" in result["response"]
        assert "4" in result["response"]
        assert "10" in result["response"]
        assert "Tails" in result["response"]
        assert len(result["sources"]) == 3  # Three tool calls


@pytest.mark.asyncio
@patch('random.choice')
async def test_agent_gets_random_fact(mock_choice, create_agent_with_sample_tools: Callable) -> None:
    """Test agent using get_random_fact tool."""
    fact = "Python was named after Monty Python, not the snake."
    mock_choice.return_value = fact
    
    agent = create_agent_with_sample_tools(
        chain=[
            "Thought: The user wants a random fact. I'll use the get_random_fact tool.\nAction: get_random_fact\nAction Input: {}",
            f"Thought: I got an interesting fact about Python.\nAnswer: Here's a fun fact: {fact}"
        ],
        tool_functions=[get_random_fact]
    )
    
    result = await agent.run("Tell me a random fact")
    
    assert "Python" in result["response"]
    assert "Monty Python" in result["response"]
    assert f"Fun fact: {fact}" in result["sources"][0]


@pytest.mark.asyncio
@patch('random.choice')
async def test_agent_tells_joke(mock_choice, create_agent_with_sample_tools: Callable) -> None:
    """Test agent using get_joke tool."""
    joke = "Why do programmers prefer dark mode? Because light attracts bugs!"
    mock_choice.return_value = joke
    
    agent = create_agent_with_sample_tools(
        chain=[
            "Thought: The user wants to hear a joke. I'll use the get_joke tool.\nAction: get_joke\nAction Input: {}",
            f"Thought: I got a programming joke.\nAnswer: Here's a joke for you: {joke}"
        ],
        tool_functions=[get_joke]
    )
    
    result = await agent.run("Tell me a programming joke")
    
    assert "dark mode" in result["response"]
    assert "bugs" in result["response"]
    assert joke in result["sources"][0]


@pytest.mark.asyncio
async def test_agent_handles_calculation_error(create_agent_with_sample_tools: Callable) -> None:
    """Test agent handling calculation errors gracefully."""
    agent = create_agent_with_sample_tools(
        chain=[
            "Thought: I'll try to calculate this expression.\nAction: calculate\nAction Input: {\"expression\": \"10 / 0\"}",
            "Thought: The calculation failed due to division by zero.\nAnswer: I cannot divide by zero - this is mathematically undefined."
        ],
        tool_functions=[calculate]
    )
    
    result = await agent.run("What is 10 divided by 0?")
    
    assert "cannot divide" in result["response"].lower() or "undefined" in result["response"].lower()
    assert "Error" in result["sources"][0]


@pytest.mark.asyncio
async def test_agent_with_all_tools_available(create_agent_with_sample_tools: Callable) -> None:
    """Test agent with all sample tools available, choosing the right one."""
    with patch('random.choice') as mock_choice:
        mock_choice.return_value = "sunny"
        
        with patch('random.randint') as mock_randint:
            mock_randint.return_value = 75
            
            agent = create_agent_with_sample_tools(
                chain=[
                    "Thought: From all available tools, I need get_weather to check the weather.\nAction: get_weather\nAction Input: {\"location\": \"Seattle\"}",
                    "Thought: The weather in Seattle is 75°F and sunny.\nAnswer: The weather in Seattle is currently 75°F and sunny."
                ],
                tool_functions=[
                    get_current_time, calculate, get_weather, search_web,
                    get_random_fact, roll_dice, flip_coin, get_joke
                ]
            )
            
            result = await agent.run("What's the weather in Seattle?")
            
            assert "75°F" in result["response"]
            assert "sunny" in result["response"]
            # Should only use weather tool despite having access to all
            assert len(result["sources"]) == 1
            assert "Weather in Seattle" in result["sources"][0]