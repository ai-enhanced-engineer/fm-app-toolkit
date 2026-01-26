"""Pydantic models for the deep research pipeline.

This module defines the data structures used across all phases
of the research pipeline: planning, gathering, synthesis, and verification.
"""

from pydantic import BaseModel, Field


class SearchStep(BaseModel):
    """A single web search step in the research plan."""

    search_terms: str = Field(description="The search query to execute")
    purpose: str = Field(description="Why this search is needed for the research")


class ResearchPlan(BaseModel):
    """Research plan output from the planning phase.

    Contains an executive summary of the research approach,
    up to 5 web search steps, and instructions for analysis.
    """

    executive_summary: str = Field(description="Brief overview of the research approach and expected outcomes")
    web_search_steps: list[SearchStep] = Field(
        default_factory=list,
        max_length=5,
        description="Ordered list of web searches to conduct (maximum 5)",
    )
    analysis_instructions: str = Field(description="Guidelines for analyzing and synthesizing the gathered information")


class SearchResult(BaseModel):
    """Result from a single web search execution."""

    query: str = Field(description="The search query that was executed")
    findings: str = Field(description="Key information extracted from search results")
    sources: list[str] = Field(default_factory=list, description="URLs or citations for the findings")


class ResearchReport(BaseModel):
    """Synthesized research report combining all findings.

    This is the main deliverable from the research pipeline.
    """

    title: str = Field(description="Descriptive title for the research report")
    summary: str = Field(description="Executive summary of key findings")
    key_findings: list[str] = Field(default_factory=list, description="Bullet-point list of main discoveries")
    sources: list[str] = Field(default_factory=list, description="All sources cited in the research")
    limitations: str = Field(description="Known limitations or gaps in the research")


class ValidationResult(BaseModel):
    """Verification result assessing research quality.

    Provides a confidence score and identifies any issues
    that should be addressed before the research is finalized.
    """

    is_valid: bool = Field(description="Whether the research meets quality standards")
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in research quality (0.0 to 1.0)",
    )
    issues_found: list[str] = Field(
        default_factory=list,
        description="List of quality issues identified",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Suggestions for improving the research",
    )
