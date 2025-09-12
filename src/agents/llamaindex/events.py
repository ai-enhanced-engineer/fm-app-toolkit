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
    """Create a StopEvent with the final workflow result."""
    return StopEvent(
        result={
            "response": response,
            "sources": sources,
            "reasoning": reasoning,
            "chat_history": chat_history,
        }
    )
