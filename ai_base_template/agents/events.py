"""Event definitions for the simple ReAct workflow.

This module defines the events used in the workflow-based ReAct agent.
These events facilitate communication between different workflow steps.
"""

from typing import Any, List

from llama_index.core.agent.react.types import BaseReasoningStep
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import ToolSelection
from llama_index.core.workflow import Event, StopEvent


class PrepEvent(Event):
    """Event to signal preparation for the next reasoning step."""
    pass


class InputEvent(Event):
    """Event containing formatted input for the LLM."""
    input: List[ChatMessage]


class ToolCallEvent(Event):
    """Event containing tool calls to execute."""
    tool_calls: List[ToolSelection]


class StreamingEvent(Event):
    """Event for streaming output chunks."""
    chunk: str


def stop_workflow(
    response: str,
    sources: List[Any],
    reasoning: List[BaseReasoningStep],
    chat_history: List[ChatMessage],
) -> StopEvent:
    """Helper function to create a StopEvent with the final result.
    
    Args:
        response: The final response text
        sources: List of sources used (e.g., tool outputs)
        reasoning: List of reasoning steps taken
        chat_history: The conversation history
        
    Returns:
        A StopEvent with the complete result
    """
    return StopEvent(
        result={
            "response": response,
            "sources": sources,
            "reasoning": reasoning,
            "chat_history": chat_history,
        }
    )