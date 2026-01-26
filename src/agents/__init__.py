"""Agents package for aiee-toolset.

This package contains agent implementations using different frameworks:
- common: Shared types and utilities (Tool dataclass)
- langgraph: LangGraph-based agents (MinimalReActAgent)
- llamaindex: LlamaIndex-based agents (SimpleReActAgent, MinimalReActAgent)
- pydantic: PydanticAI-based agents (analysis, extraction)
"""

from src.agents.common import Tool

__all__ = ["Tool"]
