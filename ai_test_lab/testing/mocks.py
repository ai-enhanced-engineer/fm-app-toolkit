"""Backward compatibility module for mock imports.

This module re-exports all mock classes from their new locations
to maintain backward compatibility with existing test imports.

All new code should import directly from the specific modules:
- ai_test_lab.testing.mock_chain for MockLLMWithChain
- ai_test_lab.testing.mock_echo for MockLLMEchoStream  
- ai_test_lab.testing.mock_rule_based for RuleBasedMockLLM
"""

from .mock_chain import MockLLMWithChain
from .mock_echo import CHUNK_SIZE, MockLLMEchoStream
from .mock_rule_based import RuleBasedMockLLM

__all__ = [
    "MockLLMWithChain",
    "MockLLMEchoStream",
    "RuleBasedMockLLM",
    "CHUNK_SIZE",
]