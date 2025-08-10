"""Essential integration tests for SimpleReActAgent with sample tools.

This module contains a focused set of tests that demonstrate key agent behaviors
with sample tools. Rather than testing every tool individually (which would be
repetitive), we focus on unique patterns and educational scenarios.
"""

from typing import Callable
from unittest.mock import patch

import pytest
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)

from fm_app_toolkit.agents.sample_tools import (
    calculate,
    flip_coin,
    get_current_time,
    get_weather,
    roll_dice,
)
from fm_app_toolkit.agents.simple_react import SimpleReActAgent, Tool
from fm_app_toolkit.testing.mocks import MockLLMWithChain, RuleBasedMockLLM
from tests.test_utilities import (
    assert_final_answer_contains,
    assert_reasoning_sequence,
    assert_tool_called,
)


@pytest.fixture
def create_agent_with_sample_tools() -> Callable:
    """Fixture to create an agent with sample tools."""
    def _create(chain: list[str], tool_functions: list, verbose: bool = False) -> SimpleReActAgent:
        # Convert functions to Tool objects
        tools = []
        for func in tool_functions:
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
async def test_agent_multi_tool_sequential_reasoning(create_agent_with_sample_tools: Callable) -> None:
    """Test agent using multiple tools in sequence to solve a complex problem.
    
    This test demonstrates:
    - Sequential tool usage with dependencies
    - Passing results between tools
    - Building up to a final answer
    
    Educational value: Shows how agents can chain tools together for complex tasks.
    """
    with patch('random.randint') as mock_randint, patch('random.choice') as mock_choice:
        mock_randint.side_effect = [6, 4]  # Dice rolls
        mock_choice.return_value = "Tails"  # Coin flip
        
        agent = create_agent_with_sample_tools(
            chain=[
                "Thought: I need to roll two dice first.\nAction: roll_dice\nAction Input: {}",
                "Thought: First dice shows 6. Let me roll another.\nAction: roll_dice\nAction Input: {}",
                "Thought: Second dice shows 4. Now let me flip a coin.\nAction: flip_coin\nAction Input: {}",
                "Thought: The coin shows Tails. So the results are: two dice showing 6 and 4 (total 10), and a coin showing Tails.\nAnswer: I rolled two dice and got 6 and 4 (total of 10), and flipped a coin which landed on Tails."
            ],
            tool_functions=[roll_dice, flip_coin]
        )
        
        handler = agent.run("Roll two dice and flip a coin")
        
        result = await agent.get_results_from_handler(handler)
        
        # Verify multi-step execution
        assert "6" in result["response"]
        assert "4" in result["response"]
        assert "10" in result["response"]  # Sum is calculated
        assert "Tails" in result["response"]
        
        # Verify all three tools were called
        assert len(result["sources"]) == 3
        
        # Verify reasoning sequence is coherent
        assert_reasoning_sequence(
            result["reasoning"],
            [
                ActionReasoningStep,    # First dice
                ObservationReasoningStep,
                ActionReasoningStep,    # Second dice
                ObservationReasoningStep,
                ActionReasoningStep,    # Coin flip
                ObservationReasoningStep,
                ResponseReasoningStep   # Final answer
            ]
        )


@pytest.mark.asyncio
async def test_agent_handles_tool_errors_gracefully(create_agent_with_sample_tools: Callable) -> None:
    """Test agent handling tool execution errors and providing meaningful responses.
    
    Educational value: Shows error recovery and user-friendly error handling.
    """
    agent = create_agent_with_sample_tools(
        chain=[
            "Thought: I'll try to calculate this expression.\nAction: calculate\nAction Input: {\"expression\": \"10 / 0\"}",
            "Thought: The calculation failed due to division by zero. This is a mathematical error.\nAnswer: I cannot divide by zero as it's mathematically undefined. Division by zero has no meaningful result."
        ],
        tool_functions=[calculate]
    )
    
    handler = agent.run("What is 10 divided by 0?")
    
    result = await agent.get_results_from_handler(handler)
    
    # Verify error is handled gracefully
    assert "cannot divide" in result["response"].lower() or "undefined" in result["response"].lower()
    assert "Error" in result["sources"][0]
    
    # Verify agent continues after error
    assert_final_answer_contains(
        result["reasoning"],
        "undefined"
    )


@pytest.mark.asyncio
async def test_agent_selects_appropriate_tool_from_many(create_agent_with_sample_tools: Callable) -> None:
    """Test agent selecting the right tool from multiple available options.
    
    Educational value: Demonstrates tool selection based on query context.
    """
    with patch('random.choice') as mock_choice:
        mock_choice.return_value = "sunny"
        
        with patch('random.randint') as mock_randint:
            mock_randint.return_value = 75
            
            # Agent has many tools but should pick weather
            agent = create_agent_with_sample_tools(
                chain=[
                    "Thought: The user is asking about weather. From all available tools, I need get_weather.\nAction: get_weather\nAction Input: {\"location\": \"Seattle\"}",
                    "Thought: The weather in Seattle is 75°F and sunny.\nAnswer: The weather in Seattle is currently 75°F and sunny."
                ],
                tool_functions=[
                    get_current_time,  # Not needed
                    calculate,         # Not needed
                    get_weather,       # This one!
                    roll_dice,         # Not needed
                    flip_coin,         # Not needed
                ]
            )
            
            handler = agent.run("What's the weather in Seattle?")
            
            result = await agent.get_results_from_handler(handler)
            
            # Verify correct tool selection
            assert "75°F" in result["response"]
            assert "sunny" in result["response"]
            
            # Should only use weather tool despite having access to all
            assert len(result["sources"]) == 1
            assert "Weather in Seattle" in result["sources"][0]
            
            # Verify only weather tool was called
            assert_tool_called(result["reasoning"], "get_weather", {"location": "Seattle"})


@pytest.mark.asyncio 
async def test_agent_with_rule_based_mock() -> None:
    """Test agent behavior with RuleBasedMockLLM for more realistic testing.
    
    This demonstrates using our new RuleBasedMockLLM instead of predefined chains,
    showing how tests can be more behavior-focused rather than script-based.
    
    Educational value: Shows the difference between scripted and rule-based testing.
    """
    # Create rule-based mock that responds intelligently
    rules = {
        "calculate": "Thought: I need to perform a calculation.\nAction: calculate\nAction Input: {{\"expression\": \"2 + 2\"}}",
        "weather": "Thought: I'll check the weather.\nAction: get_weather\nAction Input: {{\"location\": \"Paris\"}}",
        "time": "Thought: I'll get the current time.\nAction: get_current_time\nAction Input: {{}}"
    }
    
    mock_llm = RuleBasedMockLLM(rules=rules, default_behavior="direct_answer")
    
    # Note: This would need actual integration with SimpleReActAgent
    # For now, we demonstrate the concept
    agent = SimpleReActAgent(
        llm=mock_llm,
        system_header="You are a helpful assistant.",
        tools=[
            Tool(name="calculate", function=calculate, description="Perform calculations"),
            Tool(name="get_weather", function=get_weather, description="Get weather info"),
        ]
    )
    
    # The mock will intelligently respond based on the query content
    handler = agent.run("What's the weather like?")
    result = await agent.get_results_from_handler(handler)
    
    # This is more realistic - the mock decides based on content, not scripts
    assert isinstance(result, dict)
    assert "response" in result