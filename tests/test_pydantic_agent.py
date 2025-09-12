"""Tests for PydanticAI agents demonstrating key testing patterns.

This module shows how to test PydanticAI agents using TestModel for:
- Structured output validation
- Tool calling
- Dependency injection
- Both sync and async execution
"""

import pytest
from pydantic_ai.models.test import TestModel

from src.agents.pydantic import create_extraction_agent
from src.agents.pydantic.analysis_agent import AnalysisContext, AnalysisResult, create_analysis_agent
from src.agents.pydantic.extraction_agent import DataExtraction


# Test basic agent creation and structured output
@pytest.mark.asyncio
async def test_extraction_agent_basic():
    """Test basic extraction agent with structured output."""
    test_model = TestModel(
        custom_output_args={
            "entities": ["Apple", "iPhone"],
            "numbers": [394.3, 2022],
            "key_phrases": ["reported revenue", "sales growth"],
            "summary": "Apple reported strong revenue in 2022",
            "word_count": 12,
        }
    )

    agent = create_extraction_agent(model=test_model)
    result = await agent.run("Apple reported $394.3 billion in revenue for 2022, with iPhone sales leading growth.")

    assert isinstance(result.output, DataExtraction)
    assert "Apple" in result.output.entities
    assert 394.3 in result.output.numbers
    assert result.output.word_count == 12
    assert "Apple" in result.output.summary


def test_extraction_agent_sync():
    """Test synchronous execution of extraction agent."""
    test_model = TestModel(
        custom_output_args={
            "entities": ["Microsoft", "Activision Blizzard"],
            "numbers": [68.7, 2023],
            "key_phrases": ["acquisition", "billion"],
            "summary": "Microsoft acquired Activision Blizzard for $68.7 billion",
            "word_count": 10,
        }
    )

    agent = create_extraction_agent(model=test_model)
    result = agent.run_sync("Microsoft acquired Activision Blizzard for $68.7 billion in 2023.")

    assert isinstance(result.output, DataExtraction)
    assert "Microsoft" in result.output.entities
    assert 68.7 in result.output.numbers
    assert result.output.word_count == 10


@pytest.mark.asyncio
async def test_analysis_agent_with_context():
    """Test analysis agent with dependency injection."""
    test_model = TestModel(
        custom_output_args={
            "original_text": "Great product!",
            "word_count": 2,
            "sentiment": "positive",
            "confidence": 0.95,
            "key_insights": ["Customer satisfaction", "Product quality"],
        }
    )

    agent = create_analysis_agent(model=test_model)
    context = AnalysisContext(
        user_id="test_user", session_id="test_session", config={"high_precision": True}, debug=False
    )

    result = await agent.run("Great product!", deps=context)

    assert isinstance(result.output, AnalysisResult)
    assert result.output.sentiment == "positive"
    assert result.output.confidence == 0.95
    assert result.output.word_count == 2
    assert len(result.output.key_insights) == 2


@pytest.mark.asyncio
async def test_analysis_agent_debug_context():
    """Test analysis agent with debug context."""
    test_model = TestModel(
        custom_output_args={
            "original_text": "This is terrible",
            "word_count": 3,
            "sentiment": "negative",
            "confidence": 0.85,
            "key_insights": ["Customer dissatisfaction"],
        }
    )

    agent = create_analysis_agent(model=test_model)
    context = AnalysisContext(
        user_id="debug_user", session_id="debug_session", config={"high_precision": False}, debug=True
    )

    result = await agent.run("This is terrible", deps=context)

    assert isinstance(result.output, AnalysisResult)
    assert result.output.sentiment == "negative"
    assert result.output.confidence == 0.85
    assert result.output.word_count == 3


def test_extraction_agent_empty_lists():
    """Test extraction agent with empty lists in output."""
    test_model = TestModel(
        custom_output_args={
            "entities": [],
            "numbers": [],
            "key_phrases": [],
            "summary": "No significant content found",
            "word_count": 0,
        }
    )

    agent = create_extraction_agent(model=test_model)
    result = agent.run_sync("")

    assert isinstance(result.output, DataExtraction)
    assert len(result.output.entities) == 0
    assert len(result.output.numbers) == 0
    assert len(result.output.key_phrases) == 0
    assert result.output.word_count == 0


@pytest.mark.asyncio
async def test_multiple_agents_concurrency():
    """Test running multiple agents concurrently."""
    # Create different test models for different agents
    extraction_model = TestModel(
        custom_output_args={
            "entities": ["Tesla"],
            "numbers": [1000000],
            "key_phrases": ["electric vehicles"],
            "summary": "Tesla produces electric vehicles",
            "word_count": 6,
        }
    )

    analysis_model = TestModel(
        custom_output_args={
            "original_text": "Tesla is innovative",
            "word_count": 3,
            "sentiment": "positive",
            "confidence": 0.9,
            "key_insights": ["Innovation", "Technology"],
        }
    )

    extraction_agent = create_extraction_agent(model=extraction_model)
    analysis_agent = create_analysis_agent(model=analysis_model)

    context = AnalysisContext(
        user_id="concurrent_user", session_id="concurrent_session", config={"high_precision": True}, debug=False
    )

    # Run agents concurrently
    import asyncio

    extraction_task = extraction_agent.run("Tesla produces 1 million electric vehicles annually")
    analysis_task = analysis_agent.run("Tesla is innovative", deps=context)

    extraction_result, analysis_result = await asyncio.gather(extraction_task, analysis_task)

    # Verify both results
    assert isinstance(extraction_result.output, DataExtraction)
    assert "Tesla" in extraction_result.output.entities

    assert isinstance(analysis_result.output, AnalysisResult)
    assert analysis_result.output.sentiment == "positive"


@pytest.mark.asyncio
async def test_analysis_agent_high_precision_config():
    """Test analysis agent with high precision configuration."""
    test_model = TestModel(
        custom_output_args={
            "original_text": "Amazing service!",
            "word_count": 2,
            "sentiment": "positive",
            "confidence": 0.98,  # Higher confidence with high precision
            "key_insights": ["Customer satisfaction", "Service quality", "Excellence"],
        }
    )

    agent = create_analysis_agent(model=test_model)
    high_precision_context = AnalysisContext(
        user_id="precision_user", session_id="precision_session", config={"high_precision": True}, debug=False
    )

    result = await agent.run("Amazing service!", deps=high_precision_context)

    assert isinstance(result.output, AnalysisResult)
    assert result.output.confidence == 0.98
    assert len(result.output.key_insights) >= 2
