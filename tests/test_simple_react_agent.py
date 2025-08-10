"""Tests for the simple ReAct agent implementation.

These tests demonstrate how the minimalistic ReAct agent works with
mock LLMs for deterministic testing.
"""

from typing import Any, Callable

import pytest
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)

from fm_app_toolkit.agents.simple_react import SimpleReActAgent, Tool
from fm_app_toolkit.testing.mocks import MockLLMWithChain
from fm_app_toolkit.tools import add, divide, multiply, reverse_string, word_count


@pytest.fixture
def setup_simple_react_agent() -> Callable[[list[str], list[Tool] | None, int, bool], SimpleReActAgent]:
    """Fixture to create SimpleReActAgent with mock LLM."""
    def _create_agent(
        chain: list[str], 
        tools: list[Tool] | None = None, 
        max_reasoning: int = 15, 
        verbose: bool = False
    ) -> SimpleReActAgent:
        mock_llm = MockLLMWithChain(chain=chain)
        return SimpleReActAgent(
            llm=mock_llm,
            system_header="You are a helpful assistant.",
            extra_context="",
            tools=tools or [],
            max_reasoning=max_reasoning,
            verbose=verbose
        )
    return _create_agent


@pytest.mark.asyncio
async def test_single_tool_execution(setup_simple_react_agent: Callable) -> None:
    """Test agent executing a single tool to answer a query."""
    # Create agent with mock responses
    agent = setup_simple_react_agent(
        chain=[
            "Thought: I need to add 5 and 3 to get the answer.\nAction: add\nAction Input: {\"a\": 5, \"b\": 3}",
            "Thought: The addition resulted in 8.\nAnswer: The sum of 5 and 3 is 8."
        ],
        tools=[Tool(name="add", function=add, description="Add two numbers together")],
        verbose=True
    )
    
    # Run agent and get full result
    handler = agent.run(user_msg="What is 5 plus 3?")
    result: dict[str, Any] = await agent.get_results_from_handler(handler)
    
    # Verify response
    assert result["response"] == "The sum of 5 and 3 is 8."
    
    # Verify sources contain the tool output
    assert len(result["sources"]) == 1
    assert result["sources"][0] == 8
    
    # Verify we have reasoning steps
    assert len(result["reasoning"]) > 0


@pytest.mark.asyncio
async def test_multi_step_reasoning(setup_simple_react_agent: Callable) -> None:
    """Test agent performing multiple reasoning steps."""
    # Create agent with multi-step reasoning
    agent = setup_simple_react_agent(
        chain=[
            "Thought: First, I'll multiply 4 by 5.\nAction: multiply\nAction Input: {\"a\": 4, \"b\": 5}",
            "Thought: That gives us 20. Now I'll add 10.\nAction: add\nAction Input: {\"a\": 20, \"b\": 10}",
            "Thought: The final result is 30.\nAnswer: (4 × 5) + 10 = 30"
        ],
        tools=[
            Tool(name="add", function=add, description="Add two numbers"),
            Tool(name="multiply", function=multiply, description="Multiply two numbers")
        ],
        verbose=True
    )
    
    # Run agent
    handler = agent.run(user_msg="Calculate (4 * 5) + 10")
    result: dict[str, Any] = await agent.get_results_from_handler(handler)
    
    # Verify result
    assert result["response"] == "(4 × 5) + 10 = 30"
    
    # Verify both tools were used
    assert len(result["sources"]) == 2
    assert 20 in result["sources"]  # multiply result
    assert 30 in result["sources"]  # add result


@pytest.mark.asyncio
async def test_direct_answer_without_tools(setup_simple_react_agent: Callable) -> None:
    """Test agent providing direct answer without using tools."""
    # Create agent that answers directly
    agent = setup_simple_react_agent(
        chain=[
            "Thought: This is a greeting, I should respond politely.\nAnswer: Hello! I'm here to help you with calculations and text processing. How can I assist you today?"
        ],
        tools=[Tool(name="add", function=add, description="Add numbers")],
        verbose=True
    )
    
    # Run agent
    handler = agent.run(user_msg="Hello!")
    result: dict[str, Any] = await agent.get_results_from_handler(handler)
    
    # Verify we got a greeting response
    assert "Hello" in result["response"]
    assert "help" in result["response"].lower()
    
    # Verify no tools were used
    assert len(result["sources"]) == 0


@pytest.mark.asyncio
async def test_string_manipulation(setup_simple_react_agent: Callable) -> None:
    """Test agent with string manipulation tools."""
    # Create agent for string operations
    agent = setup_simple_react_agent(
        chain=[
            "Thought: I need to reverse the string 'hello'.\nAction: reverse_string\nAction Input: {\"text\": \"hello\"}",
            "Thought: The reversed string is 'olleh'.\nAnswer: The reversed version of 'hello' is 'olleh'."
        ],
        tools=[Tool(name="reverse_string", function=reverse_string, description="Reverse a string")],
        verbose=False
    )
    
    # Run agent
    handler = agent.run(user_msg="Reverse the word 'hello'")
    result: dict[str, Any] = await agent.get_results_from_handler(handler)
    
    # Verify result
    assert "olleh" in result["response"]
    assert len(result["sources"]) == 1
    assert result["sources"][0] == "olleh"


@pytest.mark.asyncio
async def test_error_handling(setup_simple_react_agent: Callable) -> None:
    """Test agent handling tool execution errors."""
    # Create agent that will trigger division by zero
    agent = setup_simple_react_agent(
        chain=[
            "Thought: I'll divide 10 by 0.\nAction: divide\nAction Input: {\"a\": 10, \"b\": 0}",
            "Thought: There was an error. Division by zero is undefined.\nAnswer: Cannot divide by zero - it's mathematically undefined."
        ],
        tools=[Tool(name="divide", function=divide, description="Divide one number by another")],
        verbose=True
    )
    
    # Run agent - should handle error gracefully
    handler = agent.run(user_msg="What is 10 divided by 0?")
    result: dict[str, Any] = await agent.get_results_from_handler(handler)
    
    # Should mention the error
    assert "zero" in result["response"].lower() or "undefined" in result["response"].lower()
    
    # Sources should be empty due to error
    assert len(result["sources"]) == 0


@pytest.mark.asyncio
async def test_max_reasoning_limit(setup_simple_react_agent: Callable) -> None:
    """Test that agent stops after max reasoning steps."""
    # Create agent with actions that exceed max_reasoning
    agent = setup_simple_react_agent(
        chain=[
            "Thought: I need to use a tool.\nAction: unknown_tool\nAction Input: {}",
            "Thought: I need to use another tool.\nAction: another_tool\nAction Input: {}",
            "Thought: Still need more.\nAction: third_tool\nAction Input: {}",
            "Thought: More actions.\nAction: fourth_tool\nAction Input: {}"
        ],
        tools=[],  # No tools, so all actions will fail but still count as reasoning steps
        max_reasoning=3,
        verbose=False
    )
    
    # Run agent
    handler = agent.run(user_msg="Complex query")
    result: dict[str, Any] = await agent.get_results_from_handler(handler)
    
    # Should return the max reasoning message
    assert "couldn't complete" in result["response"].lower() or "allowed" in result["response"].lower()
    
    # Should have hit the max reasoning limit
    assert len(result["reasoning"]) >= 3


@pytest.mark.asyncio
async def test_multiple_tools_selection(setup_simple_react_agent: Callable) -> None:
    """Test agent selecting the right tool from multiple options."""
    # Create agent that selects word_count tool
    agent = setup_simple_react_agent(
        chain=[
            "Thought: I need to count words in the text.\nAction: word_count\nAction Input: {\"text\": \"hello world test\"}",
            "Thought: There are 3 words.\nAnswer: The text 'hello world test' contains 3 words."
        ],
        tools=[
            Tool(name="add", function=add, description="Add numbers"),
            Tool(name="multiply", function=multiply, description="Multiply numbers"),
            Tool(name="word_count", function=word_count, description="Count words in text"),
            Tool(name="reverse_string", function=reverse_string, description="Reverse a string")
        ],
        verbose=True
    )
    
    # Run agent
    handler = agent.run(user_msg="How many words in 'hello world test'?")
    result: dict[str, Any] = await agent.get_results_from_handler(handler)
    
    # Verify result
    assert "3" in result["response"]
    assert len(result["sources"]) == 1
    assert result["sources"][0] == 3


def test_tool_creation() -> None:
    """Test creating tools."""
    tool = Tool(
        name="calculator",
        function=add,
        description="Adds two numbers"
    )
    
    assert tool.name == "calculator"
    assert tool.function == add
    assert tool.description == "Adds two numbers"
    
    # Test that the function works
    result = tool.function(5, 3)
    assert result == 8


def test_tool_registry() -> None:
    """Test that tools are properly registered in the agent."""
    tools = [
        Tool(name="add", function=add, description="Add"),
        Tool(name="multiply", function=multiply, description="Multiply")
    ]
    
    mock_llm = MockLLMWithChain(chain=["Answer: Test"])
    agent = SimpleReActAgent(
        llm=mock_llm, 
        system_header="You are a helpful assistant.", 
        tools=tools
    )
    
    # Check registry
    assert "add" in agent._tool_registry
    assert "multiply" in agent._tool_registry
    assert agent._tool_registry["add"].function == add
    assert agent._tool_registry["multiply"].function == multiply


@pytest.mark.asyncio
async def test_workflow_returns_dict(setup_simple_react_agent: Callable) -> None:
    """Test that the workflow returns a dictionary with expected keys."""
    agent = setup_simple_react_agent(
        chain=["Thought: Simple response.\nAnswer: Test answer"],
        tools=[]
    )
    
    # Run and get full result
    handler = agent.run(user_msg="Test")
    result: dict[str, Any] = await agent.get_results_from_handler(handler)
    
    # Check that we get a dictionary with expected keys
    assert isinstance(result, dict)
    assert "response" in result
    assert "sources" in result
    assert "reasoning" in result
    assert "chat_history" in result
    
    # Verify content
    assert result["response"] == "Test answer"


@pytest.mark.asyncio 
async def test_workflow_with_tools(setup_simple_react_agent: Callable) -> None:
    """Test workflow execution with tools."""
    agent = setup_simple_react_agent(
        chain=[
            "Thought: Adding numbers.\nAction: add\nAction Input: {\"a\": 2, \"b\": 3}",
            "Thought: Got result.\nAnswer: The sum is 5"
        ],
        tools=[Tool(name="add", function=add, description="Add numbers")]
    )
    
    handler = agent.run(user_msg="Add 2 and 3")
    
    result: dict[str, Any] = await agent.get_results_from_handler(handler)
    
    assert result["response"] == "The sum is 5"
    assert len(result["sources"]) == 1
    assert result["sources"][0] == 5


@pytest.mark.asyncio
async def test_mock_llm_chain_agent_integration_sequence(setup_simple_react_agent: Callable) -> None:
    """Validate MockLLMWithChain returns responses in exact order when used by agent.
    
    This test validates that the mock LLM properly simulates a real LLM's behavior
    by returning predefined responses in sequence, allowing deterministic testing
    of multi-step agent workflows.
    """
    # Define exact sequence of LLM responses that the agent will receive
    expected_chain = [
        "Thought: I need to multiply 2 by 3 first.\nAction: multiply\nAction Input: {\"a\": 2, \"b\": 3}",
        "Thought: Got 6. Now I'll add 4 to get the final result.\nAction: add\nAction Input: {\"a\": 6, \"b\": 4}",  
        "Thought: The calculation is complete.\nAnswer: 2 × 3 + 4 = 10"
    ]
    
    # Create agent with the chain
    agent = setup_simple_react_agent(
        chain=expected_chain,
        tools=[
            Tool(name="multiply", function=multiply, description="Multiply two numbers"),
            Tool(name="add", function=add, description="Add two numbers")
        ]
    )
    
    # Execute the agent - it should receive responses in exact order
    handler = agent.run(user_msg="Calculate 2 * 3 + 4")
    result = await agent.get_results_from_handler(handler)
    
    # Validate the agent received and processed all responses in order
    assert result["response"] == "2 × 3 + 4 = 10"
    
    # Verify tool calls happened in the expected sequence
    assert len(result["sources"]) == 2
    assert result["sources"][0] == 6  # First tool call: multiply(2, 3)
    assert result["sources"][1] == 10  # Second tool call: add(6, 4)
    
    # Validate reasoning steps match the expected sequence
    reasoning = result["reasoning"]
    assert len(reasoning) == 5  # 2 actions + 2 observations + 1 response
    
    # Verify first action (multiply)
    assert isinstance(reasoning[0], ActionReasoningStep)
    assert reasoning[0].action == "multiply"
    assert reasoning[0].action_input == {"a": 2, "b": 3}
    
    # Verify first observation
    assert isinstance(reasoning[1], ObservationReasoningStep)
    assert reasoning[1].observation == "6"
    
    # Verify second action (add)
    assert isinstance(reasoning[2], ActionReasoningStep)
    assert reasoning[2].action == "add"
    assert reasoning[2].action_input == {"a": 6, "b": 4}
    
    # Verify second observation
    assert isinstance(reasoning[3], ObservationReasoningStep)
    assert reasoning[3].observation == "10"
    
    # Verify final response
    assert isinstance(reasoning[4], ResponseReasoningStep)
    assert reasoning[4].response == "2 × 3 + 4 = 10"