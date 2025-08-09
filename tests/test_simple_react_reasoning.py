"""Comprehensive reasoning validation tests for SimpleReActAgent.

These tests validate the actual reasoning steps and their attributes,
providing deep insight into the agent's decision-making process.
"""

from typing import Any, Callable

import pytest
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)

from ai_test_lab.agents.simple_react import SimpleReActAgent, Tool
from ai_test_lab.testing.mocks import MockLLMWithChain
from ai_test_lab.tools import add, multiply


@pytest.fixture
def setup_simple_react_agent() -> Callable[[list[str], list[Tool] | None, int], SimpleReActAgent]:
    """Fixture to create SimpleReActAgent with mock LLM for reasoning tests."""
    def _create_agent(
        chain: list[str], 
        tools: list[Tool] | None = None, 
        max_reasoning: int = 1
    ) -> SimpleReActAgent:
        mock_llm = MockLLMWithChain(chain=chain)
        return SimpleReActAgent(
            llm=mock_llm,
            system_header="You are a helpful assistant that provides clear and concise answers.",
            extra_context="",
            tools=tools or [],
            max_reasoning=max_reasoning,
            verbose=False
        )
    return _create_agent


# ----------------------------------------------
# AGENT DECISION & REASONING FLOW VALIDATION
# ----------------------------------------------


@pytest.mark.asyncio
async def test_thought_produces_expected_reasoning_and_action_call(
    setup_simple_react_agent: Callable
) -> None:
    """Test that thought produces expected reasoning steps and action call."""
    agent = setup_simple_react_agent(
        chain=[
            (
                "Thought: The current language of the user is: English. I need to use a tool to help me answer the question.\n"
                "Action: add\n"
                "Action Input: {\"a\": 2, \"b\": 2}\n"
            ),
        ],
        tools=[Tool(name="add", function=add, description="Add two numbers")],
        max_reasoning=1
    )
    
    result: dict[str, Any] = await agent.run(user_msg="What is 2 + 2?")
    
    # Validate response is the max reasoning message due to limit
    assert "couldn't complete" in result["response"].lower()
    
    # Validate sources contain expected tool output
    assert len(result["sources"]) == 1
    assert result["sources"][0] == 4
    
    # Validate reasoning steps (expecting 3 steps):
    # 1. ActionReasoningStep
    # 2. ObservationReasoningStep
    # 3. ResponseReasoningStep indicating max reasoning exceeded
    reasoning_steps = result["reasoning"]
    assert len(reasoning_steps) == 3
    
    # First step: ActionReasoningStep
    first_step = reasoning_steps[0]
    assert isinstance(first_step, ActionReasoningStep)
    assert first_step.thought == "The current language of the user is: English. I need to use a tool to help me answer the question."
    assert first_step.action == "add"
    assert first_step.action_input == {"a": 2, "b": 2}
    
    # Second step: ObservationReasoningStep
    second_step = reasoning_steps[1]
    assert isinstance(second_step, ObservationReasoningStep)
    assert second_step.observation == "4"
    
    # Third step: ResponseReasoningStep indicating max reasoning exceeded
    final_step = reasoning_steps[2]
    assert isinstance(final_step, ResponseReasoningStep)
    assert "Exceeded max reasoning" in final_step.thought
    assert "couldn't complete" in final_step.response.lower()


@pytest.mark.asyncio
async def test_thought_produces_expected_reasoning_and_direct_answer(
    setup_simple_react_agent: Callable
) -> None:
    """Test direct answer without tools produces proper reasoning."""
    agent = setup_simple_react_agent(
        chain=[
            (
                "Thought: I can answer without using any more tools. I'll use the user's language to answer.\n"
                "Answer: The sum of 2 and 2 is 4.\n"
            ),
        ]
    )
    
    result: dict[str, Any] = await agent.run(user_msg="What is 2 + 2?")
    
    # Validate response
    assert result["response"] == "The sum of 2 and 2 is 4."
    
    # Validate sources should be empty (no tools used)
    assert len(result["sources"]) == 0
    
    # Validate reasoning steps
    reasoning_steps = result["reasoning"]
    assert len(reasoning_steps) == 1
    
    final_step = reasoning_steps[0]
    assert isinstance(final_step, ResponseReasoningStep)
    assert final_step.thought == "I can answer without using any more tools. I'll use the user's language to answer."
    assert final_step.response == "The sum of 2 and 2 is 4."


@pytest.mark.asyncio
async def test_thought_produces_expected_reasoning_and_no_answer(
    setup_simple_react_agent: Callable
) -> None:
    """Test response when agent cannot answer with available tools."""
    agent = setup_simple_react_agent(
        chain=[
            (
                "Thought: I cannot answer the question with the provided tools.\n"
                "Answer: Sorry, I cannot answer this question with the available tools.\n"
            ),
        ]
    )
    
    result: dict[str, Any] = await agent.run(user_msg="What is the weather?")
    
    # Validate response
    assert result["response"] == "Sorry, I cannot answer this question with the available tools."
    
    # Validate sources should be empty (no tools used)
    assert len(result["sources"]) == 0
    
    # Validate reasoning steps
    reasoning_steps = result["reasoning"]
    assert len(reasoning_steps) == 1
    
    final_step = reasoning_steps[0]
    assert isinstance(final_step, ResponseReasoningStep)
    assert final_step.thought == "I cannot answer the question with the provided tools."
    assert final_step.response == "Sorry, I cannot answer this question with the available tools."


# ----------------------------------------------
# RESPONSE HANDLING EDGE CASES
# ----------------------------------------------


@pytest.mark.asyncio
async def test_handles_empty_response_properly(
    setup_simple_react_agent: Callable
) -> None:
    """Test handling of empty LLM response."""
    agent = setup_simple_react_agent(chain=[""])  # Empty response
    
    result: dict[str, Any] = await agent.run(user_msg="Can you help me?")
    
    # With empty response, ReActOutputParser returns ResponseReasoningStep with empty response
    assert result["response"] == ""
    
    # Validate sources should be empty
    assert len(result["sources"]) == 0
    
    # Should have a ResponseReasoningStep (parser returns one for empty/malformed input)
    reasoning_steps = result["reasoning"]
    assert len(reasoning_steps) >= 1
    final_step = reasoning_steps[-1] if reasoning_steps else None
    if final_step:
        assert isinstance(final_step, ResponseReasoningStep)


@pytest.mark.asyncio
async def test_handles_untagged_output(
    setup_simple_react_agent: Callable
) -> None:
    """Test handling of untagged/malformed LLM output."""
    agent = setup_simple_react_agent(
        chain=["This is an unexpected response format that does not follow the required structure."],
        max_reasoning=2
    )
    
    result: dict[str, Any] = await agent.run(user_msg="Tell me a joke.")
    
    # ReActOutputParser handles untagged output as direct response
    assert result["response"] == "This is an unexpected response format that does not follow the required structure."
    
    # Validate sources should be empty
    assert len(result["sources"]) == 0


@pytest.mark.asyncio
async def test_handles_response_without_thought(
    setup_simple_react_agent: Callable
) -> None:
    """Test handling of Answer without Thought prefix."""
    agent = setup_simple_react_agent(
        chain=["Answer: Sure! I can help you with that."]
    )
    
    result: dict[str, Any] = await agent.run(user_msg="Can you help me?")
    
    # ReActOutputParser treats Answer without Thought as direct response
    assert result["response"] == "Answer: Sure! I can help you with that."
    
    # Validate sources should be empty
    assert len(result["sources"]) == 0
    
    # Should have a ResponseReasoningStep
    reasoning_steps = result["reasoning"]
    assert len(reasoning_steps) >= 1
    final_step = reasoning_steps[-1] if reasoning_steps else None
    if final_step:
        assert isinstance(final_step, ResponseReasoningStep)


@pytest.mark.asyncio
async def test_handles_unrecognized_tool_call(
    setup_simple_react_agent: Callable
) -> None:
    """Test handling of unknown tool call with proper reasoning."""
    agent = setup_simple_react_agent(
        chain=[
            (
                "Thought: I should use a tool to answer this.\n"
                "Action: unknown_tool\n"
                "Action Input: {\"query\": \"something\"}\n"
            ),
        ],
        tools=[],  # No tools available
        max_reasoning=1
    )
    
    result: dict[str, Any] = await agent.run(user_msg="What is the result?")
    
    # Should hit max reasoning after failed tool call
    assert "couldn't complete" in result["response"].lower()
    
    # Validate sources should be empty (tool doesn't exist)
    assert len(result["sources"]) == 0
    
    # Validate reasoning steps:
    # 1. ActionReasoningStep
    # 2. ObservationReasoningStep with error
    # 3. ResponseReasoningStep for max reasoning
    reasoning_steps = result["reasoning"]
    assert len(reasoning_steps) == 3
    
    # First step: ActionReasoningStep
    first_step = reasoning_steps[0]
    assert isinstance(first_step, ActionReasoningStep)
    assert first_step.thought == "I should use a tool to answer this."
    assert first_step.action == "unknown_tool"
    assert first_step.action_input == {"query": "something"}
    
    # Second step: ObservationReasoningStep with error
    second_step = reasoning_steps[1]
    assert isinstance(second_step, ObservationReasoningStep)
    assert "Tool 'unknown_tool' not found" in second_step.observation
    
    # Third step: ResponseReasoningStep
    third_step = reasoning_steps[2]
    assert isinstance(third_step, ResponseReasoningStep)
    assert "Exceeded max reasoning" in third_step.thought


@pytest.mark.asyncio
async def test_handles_thought_without_answer(
    setup_simple_react_agent: Callable
) -> None:
    """Test handling of Thought without Answer.
    
    When parser fails to extract action/answer from thought-only responses,
    the agent continues until either:
    1. Max reasoning is hit
    2. MockLLMWithChain runs out of messages (returns empty)
    
    In this test, we provide exactly max_reasoning thought-only responses
    so the agent will exhaust the chain and get an empty response.
    """
    agent = setup_simple_react_agent(
        chain=[
            "Thought: I need to consider various factors before answering.",
            "Thought: Still thinking about this complex question."
        ],
        max_reasoning=3  # Higher than chain length
    )
    
    result: dict[str, Any] = await agent.run(user_msg="What is the meaning of life?")
    
    # Parser will fail on first two, then get empty response from exhausted chain
    # Empty response is parsed as ResponseReasoningStep with empty content
    assert result["response"] == ""
    
    # Validate sources should be empty
    assert len(result["sources"]) == 0
    
    # Should have a ResponseReasoningStep (from empty response)
    reasoning_steps = result["reasoning"]
    assert len(reasoning_steps) >= 1
    final_step = reasoning_steps[-1] if reasoning_steps else None
    if final_step:
        assert isinstance(final_step, ResponseReasoningStep)


@pytest.mark.asyncio
async def test_max_reasoning_with_parser_errors(
    setup_simple_react_agent: Callable
) -> None:
    """Test that max reasoning limit is properly enforced even with parser errors."""
    agent = setup_simple_react_agent(
        chain=[
            "Thought: Thinking step 1",  # Parser error, doesn't count toward reasoning
            "Thought: Thinking step 2",  # Parser error, doesn't count toward reasoning  
            "Thought: Thinking step 3",  # Parser error, but hits our special handling
        ],
        max_reasoning=1  # Very low limit
    )
    
    result: dict[str, Any] = await agent.run(user_msg="Test max reasoning")
    
    # With max_reasoning=1, after first parser error we should hit the limit
    assert "couldn't complete" in result["response"].lower()
    
    # Should have ResponseReasoningStep for max reasoning
    reasoning_steps = result["reasoning"] 
    assert len(reasoning_steps) >= 1
    final_step = reasoning_steps[-1]
    assert isinstance(final_step, ResponseReasoningStep)
    assert "Exceeded max reasoning" in final_step.thought


@pytest.mark.asyncio
async def test_multi_step_reasoning_chain(
    setup_simple_react_agent: Callable
) -> None:
    """Test complete multi-step reasoning chain with multiple tools."""
    agent = setup_simple_react_agent(
        chain=[
            "Thought: First, I'll multiply 3 by 4.\nAction: multiply\nAction Input: {\"a\": 3, \"b\": 4}",
            "Thought: That gives us 12. Now I'll add 5.\nAction: add\nAction Input: {\"a\": 12, \"b\": 5}",
            "Thought: The final result is 17.\nAnswer: (3 × 4) + 5 = 17"
        ],
        tools=[
            Tool(name="add", function=add, description="Add two numbers"),
            Tool(name="multiply", function=multiply, description="Multiply two numbers")
        ],
        max_reasoning=10
    )
    
    result: dict[str, Any] = await agent.run(user_msg="Calculate (3 * 4) + 5")
    
    # Validate response
    assert result["response"] == "(3 × 4) + 5 = 17"
    
    # Validate sources
    assert len(result["sources"]) == 2
    assert 12 in result["sources"]  # multiply result
    assert 17 in result["sources"]  # add result
    
    # Validate complete reasoning chain
    reasoning_steps = result["reasoning"]
    assert len(reasoning_steps) == 5  # 2 actions + 2 observations + 1 response
    
    # First action
    assert isinstance(reasoning_steps[0], ActionReasoningStep)
    assert reasoning_steps[0].action == "multiply"
    assert reasoning_steps[0].action_input == {"a": 3, "b": 4}
    
    # First observation
    assert isinstance(reasoning_steps[1], ObservationReasoningStep)
    assert reasoning_steps[1].observation == "12"
    
    # Second action
    assert isinstance(reasoning_steps[2], ActionReasoningStep)
    assert reasoning_steps[2].action == "add"
    assert reasoning_steps[2].action_input == {"a": 12, "b": 5}
    
    # Second observation
    assert isinstance(reasoning_steps[3], ObservationReasoningStep)
    assert reasoning_steps[3].observation == "17"
    
    # Final response
    assert isinstance(reasoning_steps[4], ResponseReasoningStep)
    assert reasoning_steps[4].thought == "The final result is 17."
    assert reasoning_steps[4].response == "(3 × 4) + 5 = 17"