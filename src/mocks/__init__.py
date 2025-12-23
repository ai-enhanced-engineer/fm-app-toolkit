"""Testing utilities for LlamaIndex and LangChain applications.

This package provides mock LLM implementations for testing agent-based
applications without making real API calls.

Available Mocks:
    LlamaIndex:
        TrajectoryMockLLMLlamaIndex: Returns predefined responses in sequence
        MockLLMEchoStream: Echoes user input for streaming tests
        RuleBasedMockLLM: Generates responses based on rules and context

    LangChain:
        TrajectoryMockLLMLangChain: Returns predefined responses in sequence
"""

from .langchain import TrajectoryMockLLMLangChain
from .llamaindex import MockLLMEchoStream, RuleBasedMockLLM, TrajectoryMockLLMLlamaIndex
from .llamaindex.mock_echo import CHUNK_SIZE

__all__ = [
    "TrajectoryMockLLMLlamaIndex",
    "TrajectoryMockLLMLangChain",
    "MockLLMEchoStream",
    "RuleBasedMockLLM",
    "CHUNK_SIZE",
]
