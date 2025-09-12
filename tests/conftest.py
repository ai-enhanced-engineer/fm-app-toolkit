"""Shared test fixtures for the fm-app-toolkit test suite.

This module provides common fixtures used across multiple test files,
reducing duplication and ensuring consistency.
"""

from pathlib import Path
from typing import Callable

import pytest
from llama_index.core import Document
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

from src.agents.llamaindex.simple_react import SimpleReActAgent, Tool
from src.data_loading import LocalDocumentRepository
from src.testing.mock_chain import MockLLMWithChain
from src.testing.mock_echo import MockLLMEchoStream

# ----------------------------------------------
# PATH FIXTURES
# ----------------------------------------------


@pytest.fixture
def test_data_dir() -> Path:
    """Path to the test data directory containing sample documents."""
    return Path(__file__).parent.parent / "src" / "test_data"


# ----------------------------------------------
# MOCK FIXTURES
# ----------------------------------------------


@pytest.fixture
def mock_embed() -> MockEmbedding:
    """Standard mock embedding model for testing."""
    return MockEmbedding(embed_dim=256)


@pytest.fixture
def mock_llm_factory() -> Callable[[list[str]], MockLLMWithChain]:
    """Factory for creating MockLLMWithChain instances with custom chains."""

    def _create(chain: list[str]) -> MockLLMWithChain:
        return MockLLMWithChain(chain=chain)

    return _create


@pytest.fixture
def mock_llm_echo() -> MockLLMEchoStream:
    """Mock LLM that echoes input for testing streaming behavior."""
    return MockLLMEchoStream()


# ----------------------------------------------
# DOCUMENT FIXTURES
# ----------------------------------------------


@pytest.fixture
def sample_documents(test_data_dir: Path) -> list[Document]:
    """Load real technical documents from test data directory."""
    repo = LocalDocumentRepository(input_dir=str(test_data_dir), required_exts=[".txt"])
    return repo.load_documents(location=str(test_data_dir))


@pytest.fixture
def simple_documents() -> list[Document]:
    """Simple test documents for unit testing."""
    return [
        Document(text="First document about RAG pipelines", doc_id="1"),
        Document(text="Second document about embeddings", doc_id="2"),
        Document(text="Third document about retrieval", doc_id="3"),
    ]


# ----------------------------------------------
# AGENT FIXTURES
# ----------------------------------------------


@pytest.fixture
def create_simple_agent() -> Callable:
    """Factory for creating SimpleReActAgent with custom configuration."""

    def _create(
        chain: list[str], tools: list[Tool] | None = None, max_reasoning: int = 10, verbose: bool = False
    ) -> SimpleReActAgent:
        mock_llm = MockLLMWithChain(chain=chain)
        return SimpleReActAgent(
            llm=mock_llm,
            system_header="You are a helpful assistant.",
            tools=tools or [],
            max_reasoning=max_reasoning,
            verbose=verbose,
        )

    return _create


# ----------------------------------------------
# TOOL FIXTURES
# ----------------------------------------------


@pytest.fixture
def sample_tools() -> list[Tool]:
    """Common test tools for agent testing."""
    from src.tools import add, multiply

    return [
        Tool(name="add", fn=add, description="Add two numbers"),
        Tool(name="multiply", fn=multiply, description="Multiply two numbers"),
    ]
