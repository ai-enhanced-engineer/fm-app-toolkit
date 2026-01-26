"""Tests for the deep research pipeline demonstrating PydanticAI testing patterns.

This module provides behavioral tests for:
- Model validation (constraint enforcement, error conditions)
- Agent creation and output structure
- Error handling and graceful degradation
- Exception hierarchy
"""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from src.agents.pydantic.agents import (
    clear_agent_cache,
    create_gathering_agent,
    create_plan_agent,
    create_synthesis_agent,
    create_verification_agent,
)
from src.agents.pydantic.exceptions import GatheringError, ResearchPipelineError
from src.agents.pydantic.models import (
    ResearchPlan,
    ResearchReport,
    SearchResult,
    SearchStep,
    ValidationResult,
)
from src.agents.pydantic.research import _phase_gathering, _save_results

# ============================================================================
# Model Constraint Tests - Verify business rules are enforced
# ============================================================================


class TestSearchStepConstraints:
    """Tests for SearchStep model constraints."""

    def test__search_step__rejects_missing_search_terms(self) -> None:
        """Test that SearchStep requires search_terms field."""
        with pytest.raises(ValidationError) as exc_info:
            SearchStep(purpose="Learn basic patterns")  # type: ignore[call-arg]
        assert "search_terms" in str(exc_info.value)

    def test__search_step__rejects_missing_purpose(self) -> None:
        """Test that SearchStep requires purpose field."""
        with pytest.raises(ValidationError) as exc_info:
            SearchStep(search_terms="PydanticAI tutorial")  # type: ignore[call-arg]
        assert "purpose" in str(exc_info.value)


class TestResearchPlanConstraints:
    """Tests for ResearchPlan model constraints."""

    def test__research_plan__enforces_max_five_steps(self) -> None:
        """Test that ResearchPlan rejects more than 5 search steps."""
        steps = [SearchStep(search_terms=f"query {i}", purpose=f"purpose {i}") for i in range(6)]
        with pytest.raises(ValidationError) as exc_info:
            ResearchPlan(
                executive_summary="Summary",
                web_search_steps=steps,
                analysis_instructions="Instructions",
            )
        # Verify the error is about the list length constraint
        assert "web_search_steps" in str(exc_info.value)

    def test__research_plan__accepts_exactly_five_steps(self) -> None:
        """Test that ResearchPlan accepts exactly 5 search steps (boundary)."""
        steps = [SearchStep(search_terms=f"query {i}", purpose=f"purpose {i}") for i in range(5)]
        plan = ResearchPlan(
            executive_summary="Summary",
            web_search_steps=steps,
            analysis_instructions="Instructions",
        )
        assert len(plan.web_search_steps) == 5

    def test__research_plan__accepts_zero_steps(self) -> None:
        """Test that ResearchPlan accepts empty search steps."""
        plan = ResearchPlan(
            executive_summary="Summary",
            web_search_steps=[],
            analysis_instructions="Instructions",
        )
        assert len(plan.web_search_steps) == 0


class TestValidationResultConstraints:
    """Tests for ValidationResult model constraints."""

    def test__validation_result__rejects_confidence_above_one(self) -> None:
        """Test that confidence_score must be <= 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            ValidationResult(
                is_valid=True,
                confidence_score=1.01,
                issues_found=[],
                recommendations=[],
            )
        assert "confidence_score" in str(exc_info.value)

    def test__validation_result__rejects_negative_confidence(self) -> None:
        """Test that confidence_score must be >= 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            ValidationResult(
                is_valid=True,
                confidence_score=-0.01,
                issues_found=[],
                recommendations=[],
            )
        assert "confidence_score" in str(exc_info.value)

    def test__validation_result__accepts_boundary_values(self) -> None:
        """Test that confidence_score accepts 0.0 and 1.0 (boundaries)."""
        result_min = ValidationResult(
            is_valid=False,
            confidence_score=0.0,
            issues_found=["Low quality"],
            recommendations=[],
        )
        result_max = ValidationResult(
            is_valid=True,
            confidence_score=1.0,
            issues_found=[],
            recommendations=[],
        )
        assert result_min.confidence_score == 0.0
        assert result_max.confidence_score == 1.0


# ============================================================================
# Agent Creation Tests - Verify agents are configured correctly
# ============================================================================


class TestPlanAgentBehavior:
    """Tests for the planning agent behavior."""

    @pytest.mark.asyncio
    async def test__plan_agent__returns_structured_research_plan(self) -> None:
        """Test that plan agent returns a properly structured ResearchPlan."""
        test_model = TestModel(
            custom_output_args={
                "executive_summary": "Research approach for PydanticAI",
                "web_search_steps": [
                    {"search_terms": "PydanticAI documentation", "purpose": "Official docs"},
                    {"search_terms": "PydanticAI examples", "purpose": "Usage patterns"},
                ],
                "analysis_instructions": "Focus on practical applications",
            }
        )

        agent = create_plan_agent(model=test_model)
        result = await agent.run("Research PydanticAI features")

        # Verify the output is the correct type
        assert isinstance(result.output, ResearchPlan)
        # Verify nested objects are properly constructed
        assert all(isinstance(step, SearchStep) for step in result.output.web_search_steps)


class TestGatheringAgentConfiguration:
    """Tests for the gathering agent configuration."""

    def test__gathering_agent__has_web_search_builtin_tool(self) -> None:
        """Test that gathering agent is configured with WebSearchTool."""
        from pydantic_ai import WebSearchTool

        # Use TestModel to avoid requiring real API keys on CI
        test_model = TestModel(
            custom_output_args={
                "query": "test query",
                "findings": "test findings",
                "sources": ["https://example.com"],
            }
        )
        agent = create_gathering_agent(model=test_model)

        # Verify WebSearchTool is in builtin_tools
        assert agent._builtin_tools is not None
        assert len(agent._builtin_tools) == 1
        assert isinstance(agent._builtin_tools[0], WebSearchTool)


class TestSynthesisAgentBehavior:
    """Tests for the synthesis agent behavior."""

    @pytest.mark.asyncio
    async def test__synthesis_agent__returns_structured_report(self) -> None:
        """Test that synthesis agent returns a properly structured ResearchReport."""
        test_model = TestModel(
            custom_output_args={
                "title": "PydanticAI Features Report",
                "summary": "Comprehensive analysis of PydanticAI capabilities",
                "key_findings": ["Type-safe outputs", "Built-in validation"],
                "sources": ["https://ai.pydantic.dev"],
                "limitations": "Limited to English documentation",
            }
        )

        agent = create_synthesis_agent(model=test_model)
        result = await agent.run("Synthesize findings about PydanticAI")

        # Verify the output is the correct type
        assert isinstance(result.output, ResearchReport)
        # Verify lists are properly populated
        assert isinstance(result.output.key_findings, list)
        assert isinstance(result.output.sources, list)


class TestVerificationAgentBehavior:
    """Tests for the verification agent behavior."""

    @pytest.mark.asyncio
    async def test__verification_agent__returns_validation_with_score(self) -> None:
        """Test that verification agent returns ValidationResult with valid score."""
        test_model = TestModel(
            custom_output_args={
                "is_valid": True,
                "confidence_score": 0.85,
                "issues_found": [],
                "recommendations": ["Consider adding more recent sources"],
            }
        )

        agent = create_verification_agent(model=test_model)
        result = await agent.run("Verify this research report")

        # Verify the output is the correct type
        assert isinstance(result.output, ValidationResult)
        # Verify confidence score is within valid range
        assert 0.0 <= result.output.confidence_score <= 1.0


# ============================================================================
# Error Handling Tests - Verify graceful degradation
# ============================================================================


class TestGatheringPhaseErrorHandling:
    """Tests for error handling in the gathering phase."""

    @pytest.mark.asyncio
    async def test__phase_gathering__handles_partial_failures(self) -> None:
        """Test that gathering phase continues despite some search failures."""
        plan = ResearchPlan(
            executive_summary="Test plan",
            web_search_steps=[
                SearchStep(search_terms="query 1", purpose="purpose 1"),
                SearchStep(search_terms="query 2", purpose="purpose 2"),
                SearchStep(search_terms="query 3", purpose="purpose 3"),
            ],
            analysis_instructions="Test instructions",
        )

        # Mock _execute_single_search to return mix of success and failure
        async def mock_search(query: str, purpose: str) -> SearchResult:
            if "query 2" in query:
                raise RuntimeError("Network error")
            return SearchResult(query=query, findings="Found info", sources=["http://example.com"])

        with patch("src.agents.pydantic.research._execute_single_search", side_effect=mock_search):
            results = await _phase_gathering(plan)

        # Should have 2 successful results (query 1 and query 3)
        assert len(results) == 2
        queries = [r.query for r in results]
        assert "query 1" in queries
        assert "query 3" in queries

    @pytest.mark.asyncio
    async def test__phase_gathering__returns_empty_list_when_all_fail(self) -> None:
        """Test that gathering phase returns empty list when all searches fail."""
        plan = ResearchPlan(
            executive_summary="Test plan",
            web_search_steps=[
                SearchStep(search_terms="query 1", purpose="purpose 1"),
                SearchStep(search_terms="query 2", purpose="purpose 2"),
            ],
            analysis_instructions="Test instructions",
        )

        # Mock all searches to fail
        async def mock_search(query: str, purpose: str) -> SearchResult:
            raise RuntimeError("All searches fail")

        with patch("src.agents.pydantic.research._execute_single_search", side_effect=mock_search):
            results = await _phase_gathering(plan)

        # Should have 0 results
        assert len(results) == 0


class TestExceptionHierarchy:
    """Tests for the custom exception hierarchy."""

    def test__gathering_error__is_research_pipeline_error(self) -> None:
        """Test that GatheringError inherits from ResearchPipelineError."""
        error = GatheringError(attempted=5, failed=5)
        assert isinstance(error, ResearchPipelineError)
        assert isinstance(error, Exception)

    def test__gathering_error__contains_metadata(self) -> None:
        """Test that GatheringError contains useful metadata."""
        error = GatheringError(attempted=5, failed=3)
        assert error.attempted == 5
        assert error.failed == 3
        assert "5" in str(error)  # Message includes attempted count

    def test__gathering_error__can_be_caught_specifically(self) -> None:
        """Test that GatheringError can be caught specifically."""
        try:
            raise GatheringError(attempted=3, failed=3)
        except GatheringError as e:
            assert e.attempted == 3
        except Exception:
            pytest.fail("Should have caught GatheringError specifically")


# ============================================================================
# File Output Tests - Verify save functionality
# ============================================================================


class TestSaveResults:
    """Tests for the _save_results function."""

    def test__save_results__creates_json_and_md_files(self, tmp_path: Path) -> None:
        """Test that _save_results creates both JSON and Markdown files."""
        report = ResearchReport(
            title="Test Report",
            summary="Test summary",
            key_findings=["Finding 1", "Finding 2"],
            sources=["https://example.com"],
            limitations="Test limitations",
        )
        validation = ValidationResult(
            is_valid=True,
            confidence_score=0.9,
            issues_found=[],
            recommendations=["Recommendation 1"],
        )

        json_path, md_path = _save_results("Test Topic", report, validation, tmp_path)

        # Verify files exist
        assert json_path.exists()
        assert md_path.exists()
        assert json_path.suffix == ".json"
        assert md_path.suffix == ".md"

    def test__save_results__handles_special_characters_in_topic(self, tmp_path: Path) -> None:
        """Test that _save_results sanitizes special characters in topic."""
        report = ResearchReport(
            title="Test",
            summary="Test",
            key_findings=[],
            sources=[],
            limitations="None",
        )
        validation = ValidationResult(
            is_valid=True,
            confidence_score=0.5,
            issues_found=[],
            recommendations=[],
        )

        # Topic with special characters
        json_path, _ = _save_results("Test!@#$%^&*()Topic", report, validation, tmp_path)

        # Filename should be sanitized (no special chars except underscore)
        filename = json_path.stem  # Get filename without extension
        assert "!" not in filename
        assert "@" not in filename
        assert "#" not in filename

    def test__save_results__handles_empty_topic(self, tmp_path: Path) -> None:
        """Test that _save_results handles topics that sanitize to empty."""
        report = ResearchReport(
            title="Test",
            summary="Test",
            key_findings=[],
            sources=[],
            limitations="None",
        )
        validation = ValidationResult(
            is_valid=True,
            confidence_score=0.5,
            issues_found=[],
            recommendations=[],
        )

        # Topic with only special characters
        json_path, _ = _save_results("!@#$%", report, validation, tmp_path)

        # Should fall back to "research" slug
        assert "research" in json_path.stem


# ============================================================================
# Concurrent Execution Tests - Verify parallel behavior
# ============================================================================


class TestConcurrentAgents:
    """Tests for concurrent agent execution."""

    @pytest.mark.asyncio
    async def test__multiple_agents__run_concurrently(self) -> None:
        """Test that multiple agents can run in parallel."""
        test_model = TestModel(
            custom_output_args={
                "title": "Test Report",
                "summary": "Test summary",
                "key_findings": ["Finding 1"],
                "sources": ["https://example.com"],
                "limitations": "Test limitations",
            }
        )

        agent = create_synthesis_agent(model=test_model)

        # Run 3 tasks concurrently
        tasks = [agent.run(f"Synthesize query {i}") for i in range(3)]
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result.output, ResearchReport)


# ============================================================================
# Agent Cache Tests - Verify caching behavior
# ============================================================================


class TestAgentCaching:
    """Tests for agent caching behavior."""

    def test__clear_agent_cache__clears_all_caches(self) -> None:
        """Test that clear_agent_cache actually clears cached agents."""
        from src.agents.pydantic.agents import (
            get_gathering_agent,
            get_plan_agent,
            get_synthesis_agent,
            get_verification_agent,
        )

        # First clear any existing cache state
        clear_agent_cache()

        # Verify caches start empty
        assert get_plan_agent.cache_info().currsize == 0
        assert get_gathering_agent.cache_info().currsize == 0
        assert get_synthesis_agent.cache_info().currsize == 0
        assert get_verification_agent.cache_info().currsize == 0

        # Mock the create functions to avoid needing API keys
        test_model = TestModel(
            custom_output_args={
                "executive_summary": "test",
                "web_search_steps": [],
                "analysis_instructions": "test",
            }
        )

        with (
            patch("src.agents.pydantic.agents.create_plan_agent", return_value=test_model),
            patch("src.agents.pydantic.agents.create_gathering_agent", return_value=test_model),
            patch("src.agents.pydantic.agents.create_synthesis_agent", return_value=test_model),
            patch("src.agents.pydantic.agents.create_verification_agent", return_value=test_model),
        ):
            # Populate caches by calling getters
            get_plan_agent()
            get_gathering_agent()
            get_synthesis_agent()
            get_verification_agent()

            # Verify caches have items
            assert get_plan_agent.cache_info().currsize > 0
            assert get_gathering_agent.cache_info().currsize > 0
            assert get_synthesis_agent.cache_info().currsize > 0
            assert get_verification_agent.cache_info().currsize > 0

        # Clear caches
        clear_agent_cache()

        # Verify caches are empty
        assert get_plan_agent.cache_info().currsize == 0
        assert get_gathering_agent.cache_info().currsize == 0
        assert get_synthesis_agent.cache_info().currsize == 0
        assert get_verification_agent.cache_info().currsize == 0
