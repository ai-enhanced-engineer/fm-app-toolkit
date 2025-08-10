"""Test utilities for more intelligent and pedagogical testing.

This module provides utilities that enable behavior-based testing rather than
simple mock-response validation. These tools help create tests that actually
validate agent behavior and provide educational value.
"""

from typing import Any, Dict, List, Optional, Type

from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)


def assert_reasoning_sequence(
    reasoning_steps: List[BaseReasoningStep], expected_sequence: List[Type[BaseReasoningStep]]
) -> None:
    """Assert that reasoning steps follow expected sequence of types.

    Args:
        reasoning_steps: Actual reasoning steps from agent
        expected_sequence: Expected sequence of step types

    Raises:
        AssertionError: If sequence doesn't match
    """
    assert len(reasoning_steps) == len(expected_sequence), (
        f"Expected {len(expected_sequence)} steps, got {len(reasoning_steps)}"
    )

    for i, (step, expected_type) in enumerate(zip(reasoning_steps, expected_sequence)):
        assert isinstance(step, expected_type), (
            f"Step {i}: Expected {expected_type.__name__}, got {type(step).__name__}"
        )


def assert_tool_called(
    reasoning_steps: List[BaseReasoningStep], tool_name: str, expected_input: Optional[Dict[str, Any]] = None
) -> None:
    """Assert that a specific tool was called with expected input.

    Args:
        reasoning_steps: Reasoning steps from agent
        tool_name: Name of tool that should have been called
        expected_input: Expected input to tool (if provided)

    Raises:
        AssertionError: If tool wasn't called or input doesn't match
    """
    tool_calls = [
        step for step in reasoning_steps if isinstance(step, ActionReasoningStep) and step.action == tool_name
    ]

    assert tool_calls, f"Tool '{tool_name}' was not called"

    if expected_input is not None:
        actual_input = tool_calls[0].action_input
        assert actual_input == expected_input, f"Tool input mismatch. Expected: {expected_input}, Got: {actual_input}"


def assert_final_answer_contains(reasoning_steps: List[BaseReasoningStep], expected_content: str) -> None:
    """Assert that final answer contains expected content.

    Args:
        reasoning_steps: Reasoning steps from agent
        expected_content: Content that should appear in final answer

    Raises:
        AssertionError: If no final answer or content not found
    """
    response_steps = [step for step in reasoning_steps if isinstance(step, ResponseReasoningStep)]

    assert response_steps, "No final answer found in reasoning steps"

    final_response = response_steps[-1].response
    assert expected_content.lower() in final_response.lower(), (
        f"Expected '{expected_content}' in response, got: {final_response}"
    )


def create_test_tools_dict() -> Dict[str, Any]:
    """Create a standard set of test tools for consistency.

    Returns:
        Dictionary of tool name to tool function
    """

    def add(a: int, b: int) -> int:
        return a + b

    def multiply(a: int, b: int) -> int:
        return a * b

    def get_weather(location: str) -> str:
        return f"Weather in {location}: 72°F and sunny"

    return {
        "add": add,
        "multiply": multiply,
        "get_weather": get_weather,
    }


def validate_reasoning_chain_coherence(reasoning_steps: List[BaseReasoningStep]) -> bool:
    """Validate that a reasoning chain is coherent and well-formed.

    A coherent chain should:
    - Have alternating Action/Observation pairs
    - End with a Response (unless interrupted)
    - Have observations for all actions

    Args:
        reasoning_steps: Steps to validate

    Returns:
        True if chain is coherent
    """
    if not reasoning_steps:
        return False

    # Check for proper alternation
    expecting_observation = False
    for step in reasoning_steps:
        if isinstance(step, ActionReasoningStep):
            if expecting_observation:
                return False  # Got action when expecting observation
            expecting_observation = True
        elif isinstance(step, ObservationReasoningStep):
            if not expecting_observation:
                return False  # Got observation without action
            expecting_observation = False
        elif isinstance(step, ResponseReasoningStep):
            # Response can come at any time as final step
            break

    # If we ended expecting an observation, that's invalid
    # (unless the last step is a ResponseReasoningStep)
    if expecting_observation and not isinstance(reasoning_steps[-1], ResponseReasoningStep):
        return False

    return True
