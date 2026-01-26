"""Agent factories for the deep research pipeline.

This module provides agent factories for each phase of the research pipeline:
- Planning: Creates research plans with up to 5 search steps
- Gathering: Executes web searches with WebSearchTool
- Synthesis: Combines findings into coherent reports
- Verification: Validates research quality and provides confidence scores

Cached versions use @lru_cache for singleton-like behavior within a process.
Uncached versions (create_*_agent) allow passing custom model objects for testing.

Environment Variables:
    RESEARCH_PLAN_MODEL: Model for planning agent (default: anthropic:claude-sonnet-4-20250514)
    RESEARCH_GATHERING_MODEL: Model for gathering agent (default: google-gla:gemini-2.0-flash)
    RESEARCH_SYNTHESIS_MODEL: Model for synthesis agent (default: anthropic:claude-sonnet-4-20250514)
    RESEARCH_VERIFICATION_MODEL: Model for verification agent (default: anthropic:claude-sonnet-4-20250514)
"""

import os
from functools import lru_cache
from typing import Any

from pydantic_ai import Agent, WebSearchTool

from src.logging import get_logger

from .models import ResearchPlan, ResearchReport, SearchResult, ValidationResult

logger = get_logger(__name__)

# Default models (can be overridden via environment variables)
DEFAULT_PLAN_MODEL = os.getenv("RESEARCH_PLAN_MODEL", "anthropic:claude-sonnet-4-20250514")
DEFAULT_GATHERING_MODEL = os.getenv("RESEARCH_GATHERING_MODEL", "google-gla:gemini-2.0-flash")
DEFAULT_SYNTHESIS_MODEL = os.getenv("RESEARCH_SYNTHESIS_MODEL", "anthropic:claude-sonnet-4-20250514")
DEFAULT_VERIFICATION_MODEL = os.getenv("RESEARCH_VERIFICATION_MODEL", "anthropic:claude-sonnet-4-20250514")


def create_plan_agent(model: Any = DEFAULT_PLAN_MODEL) -> Agent[None, ResearchPlan]:
    """Create a research planning agent (uncached).

    Use this for testing with TestModel or when you need a fresh agent instance.

    Args:
        model: The LLM model specification or model object.

    Returns:
        Configured PydanticAI agent for research planning.
    """
    logger.info("creating_plan_agent", model=str(model))

    return Agent(
        model,
        output_type=ResearchPlan,
        system_prompt="""You are a research planning expert. Your task is to create
comprehensive research plans for investigating topics thoroughly.

When given a research topic:
1. Analyze what information is needed to fully understand the topic
2. Create an executive summary of your research approach
3. Design up to 5 focused web searches that will gather diverse, high-quality information
4. Provide clear instructions for how to analyze and synthesize the findings

Each search step should:
- Have specific, targeted search terms
- Serve a distinct purpose in the overall research
- Build on or complement other searches

Prioritize authoritative sources and recent information.""",
        instrument=True,
    )


def create_gathering_agent(model: Any = DEFAULT_GATHERING_MODEL) -> Agent[None, SearchResult]:
    """Create a web search gathering agent (uncached).

    Use this for testing with TestModel or when you need a fresh agent instance.

    Args:
        model: The LLM model specification or model object.

    Returns:
        Configured PydanticAI agent with web search capability.
    """
    logger.info("creating_gathering_agent", model=str(model))

    return Agent(
        model,
        output_type=SearchResult,
        builtin_tools=[WebSearchTool()],
        system_prompt="""You are a research assistant specialized in web search and information extraction.

When given a search query and purpose:
1. Execute the web search using the provided tool
2. Analyze the search results for relevant information
3. Extract key findings that address the research purpose
4. Note all sources (URLs) for citation

Focus on:
- Factual, verifiable information
- Recent and authoritative sources
- Information directly relevant to the stated purpose
- Diverse perspectives when applicable

Always cite your sources accurately.""",
        instrument=True,
    )


def create_synthesis_agent(model: Any = DEFAULT_SYNTHESIS_MODEL) -> Agent[None, ResearchReport]:
    """Create a research synthesis agent (uncached).

    Use this for testing with TestModel or when you need a fresh agent instance.

    Args:
        model: The LLM model specification or model object.

    Returns:
        Configured PydanticAI agent for report synthesis.
    """
    logger.info("creating_synthesis_agent", model=str(model))

    return Agent(
        model,
        output_type=ResearchReport,
        system_prompt="""You are a research synthesis expert. Your task is to combine
multiple search results into a comprehensive, coherent research report.

When synthesizing research:
1. Create a descriptive, informative title
2. Write a clear executive summary of key findings
3. Extract and organize key findings as distinct bullet points
4. Compile all sources from the search results
5. Honestly assess limitations and gaps in the research

Your report should:
- Present information logically and clearly
- Avoid redundancy while being thorough
- Distinguish between facts and interpretations
- Acknowledge conflicting information when present
- Be useful for someone who hasn't seen the raw search results""",
        instrument=True,
    )


def create_verification_agent(model: Any = DEFAULT_VERIFICATION_MODEL) -> Agent[None, ValidationResult]:
    """Create a research verification agent (uncached).

    Use this for testing with TestModel or when you need a fresh agent instance.

    Args:
        model: The LLM model specification or model object.

    Returns:
        Configured PydanticAI agent for quality verification.
    """
    logger.info("creating_verification_agent", model=str(model))

    return Agent(
        model,
        output_type=ValidationResult,
        system_prompt="""You are a research quality verification expert. Your task is to
assess research reports for accuracy, completeness, and reliability.

When verifying research:
1. Check if the report adequately addresses the original research topic
2. Assess the quality and diversity of sources
3. Look for logical consistency in the findings
4. Identify any gaps, biases, or unsupported claims
5. Evaluate the overall reliability of the research

Provide:
- is_valid: True if the research meets basic quality standards
- confidence_score: 0.0-1.0 reflecting your confidence in the research quality
  - 0.9-1.0: Excellent research with authoritative sources
  - 0.7-0.9: Good research with minor gaps
  - 0.5-0.7: Adequate research with notable limitations
  - Below 0.5: Research needs significant improvement
- issues_found: Specific problems identified
- recommendations: Actionable suggestions for improvement

Be constructive but honest in your assessment.""",
        instrument=True,
    )


# Cached versions for production use
@lru_cache(maxsize=1)
def get_plan_agent(model: str = DEFAULT_PLAN_MODEL) -> Agent[None, ResearchPlan]:
    """Get a cached research planning agent.

    Args:
        model: The LLM model specification string.

    Returns:
        Cached PydanticAI agent for research planning.
    """
    return create_plan_agent(model)


@lru_cache(maxsize=1)
def get_gathering_agent(model: str = DEFAULT_GATHERING_MODEL) -> Agent[None, SearchResult]:
    """Get a cached web search gathering agent.

    Args:
        model: The LLM model specification string.

    Returns:
        Cached PydanticAI agent with web search capability.
    """
    return create_gathering_agent(model)


@lru_cache(maxsize=1)
def get_synthesis_agent(model: str = DEFAULT_SYNTHESIS_MODEL) -> Agent[None, ResearchReport]:
    """Get a cached research synthesis agent.

    Args:
        model: The LLM model specification string.

    Returns:
        Cached PydanticAI agent for report synthesis.
    """
    return create_synthesis_agent(model)


@lru_cache(maxsize=1)
def get_verification_agent(model: str = DEFAULT_VERIFICATION_MODEL) -> Agent[None, ValidationResult]:
    """Get a cached research verification agent.

    Args:
        model: The LLM model specification string.

    Returns:
        Cached PydanticAI agent for quality verification.
    """
    return create_verification_agent(model)


def clear_agent_cache() -> None:
    """Clear all cached agents.

    Useful for testing or when reconfiguration is needed.
    """
    get_plan_agent.cache_clear()
    get_gathering_agent.cache_clear()
    get_synthesis_agent.cache_clear()
    get_verification_agent.cache_clear()
    logger.info("agent_cache_cleared")
