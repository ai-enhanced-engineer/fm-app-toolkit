"""Event definitions for the simple ReAct workflow.

This module defines helper functions for the ReAct agent.
Since we now inherit from BaseWorkflowAgent, we use its built-in events.
"""

from typing import Any, List

from llama_index.core.agent.react.types import BaseReasoningStep
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.workflow import StopEvent


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
