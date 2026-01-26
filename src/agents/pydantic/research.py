"""Deep research orchestration with CLI entry point.

This module provides the main orchestration for the 4-phase deep research pipeline:
1. Planning - Create research plan with up to 5 search steps
2. Gathering - Execute web searches in parallel
3. Synthesis - Combine findings into a coherent report
4. Verification - Validate research quality

Usage:
    python -m src.agents.pydantic.research --topic "Your research topic"
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.logging import get_logger

from .agents import (
    get_gathering_agent,
    get_plan_agent,
    get_synthesis_agent,
    get_verification_agent,
)
from .exceptions import GatheringError
from .models import ResearchPlan, ResearchReport, SearchResult, ValidationResult

logger = get_logger(__name__)

# Default output directory for research results
DEFAULT_OUTPUT_DIR = Path("output/research")


async def _phase_planning(topic: str) -> ResearchPlan:
    """Phase 1: Create research plan."""
    logger.info("phase_planning_start", topic=topic)

    agent = get_plan_agent()
    result = await agent.run(f"Create a research plan for: {topic}")

    logger.info(
        "phase_planning_complete",
        search_steps=len(result.output.web_search_steps),
    )
    return result.output


async def _execute_single_search(query: str, purpose: str) -> SearchResult:
    """Execute a single web search."""
    agent = get_gathering_agent()
    prompt = f"""Execute a web search and extract findings.

Search query: {query}
Purpose: {purpose}

Use the web search tool to gather information, then summarize the findings."""

    result = await agent.run(prompt)
    return result.output


async def _phase_gathering(plan: ResearchPlan) -> list[SearchResult]:
    """Phase 2: Execute web searches in parallel."""
    logger.info(
        "phase_gathering_start",
        num_searches=len(plan.web_search_steps),
    )

    # Execute all searches in parallel
    tasks = [_execute_single_search(step.search_terms, step.purpose) for step in plan.web_search_steps]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and log them
    search_results: list[SearchResult] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning(
                "search_failed",
                search_index=i,
                error=str(result),
            )
        elif isinstance(result, SearchResult):
            search_results.append(result)

    logger.info(
        "phase_gathering_complete",
        successful_searches=len(search_results),
        failed_searches=len(results) - len(search_results),
    )
    return search_results


async def _phase_synthesis(
    topic: str,
    plan: ResearchPlan,
    search_results: list[SearchResult],
) -> ResearchReport:
    """Phase 3: Synthesize findings into a report."""
    logger.info("phase_synthesis_start", num_results=len(search_results))

    agent = get_synthesis_agent()

    # Format search results for synthesis
    findings_text = "\n\n".join(
        [
            f"## Search: {result.query}\nFindings: {result.findings}\nSources: {', '.join(result.sources)}"
            for result in search_results
        ]
    )

    prompt = f"""Synthesize the following research findings into a comprehensive report.

Original Topic: {topic}

Research Approach: {plan.executive_summary}

Analysis Instructions: {plan.analysis_instructions}

Search Findings:
{findings_text}

Create a well-structured research report based on these findings."""

    result = await agent.run(prompt)

    logger.info("phase_synthesis_complete", title=result.output.title)
    return result.output


async def _phase_verification(
    topic: str,
    report: ResearchReport,
) -> ValidationResult:
    """Phase 4: Verify research quality."""
    logger.info("phase_verification_start")

    agent = get_verification_agent()

    prompt = f"""Verify the quality of this research report.

Original Topic: {topic}

Report Title: {report.title}

Summary: {report.summary}

Key Findings:
{chr(10).join(f"- {finding}" for finding in report.key_findings)}

Sources: {", ".join(report.sources)}

Limitations: {report.limitations}

Assess the quality, completeness, and reliability of this research."""

    result = await agent.run(prompt)

    logger.info(
        "phase_verification_complete",
        is_valid=result.output.is_valid,
        confidence=result.output.confidence_score,
    )
    return result.output


def _save_results(
    topic: str,
    report: ResearchReport,
    validation: ValidationResult,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save research results to JSON and Markdown files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename-safe topic slug
    slug = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)
    slug = slug.replace(" ", "_")[:50].lower().strip("_")
    # Fallback to "research" if slug is empty after sanitization
    if not slug:
        slug = "research"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{slug}_{timestamp}"

    # Save JSON
    json_path = output_dir / f"{base_name}.json"
    json_data = {
        "topic": topic,
        "report": report.model_dump(),
        "validation": validation.model_dump(),
        "generated_at": datetime.now().isoformat(),
    }
    json_path.write_text(json.dumps(json_data, indent=2))

    # Save Markdown
    md_path = output_dir / f"{base_name}.md"
    md_content = f"""# {report.title}

**Research Topic:** {topic}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Confidence Score:** {validation.confidence_score:.2f}

---

## Summary

{report.summary}

## Key Findings

{chr(10).join(f"- {finding}" for finding in report.key_findings)}

## Limitations

{report.limitations}

## Sources

{chr(10).join(f"- {source}" for source in report.sources)}

---

## Quality Assessment

**Valid:** {"Yes" if validation.is_valid else "No"}

**Issues Found:**
{chr(10).join(f"- {issue}" for issue in validation.issues_found) if validation.issues_found else "None"}

**Recommendations:**
{chr(10).join(f"- {rec}" for rec in validation.recommendations) if validation.recommendations else "None"}
"""
    md_path.write_text(md_content)

    return json_path, md_path


async def run_research(
    topic: str,
    output_dir: Path | None = None,
) -> tuple[ResearchReport, ValidationResult]:
    """Run the complete deep research pipeline.

    Args:
        topic: The research topic to investigate.
        output_dir: Directory for saving results. Defaults to output/research/.

    Returns:
        Tuple of (ResearchReport, ValidationResult).
    """
    logger.info("research_pipeline_start", topic=topic)

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    # Phase 1: Planning
    plan = await _phase_planning(topic)

    # Phase 2: Gathering (parallel execution)
    search_results = await _phase_gathering(plan)

    if not search_results:
        raise GatheringError(
            attempted=len(plan.web_search_steps),
            failed=len(plan.web_search_steps),
        )

    # Phase 3: Synthesis
    report = await _phase_synthesis(topic, plan, search_results)

    # Phase 4: Verification
    validation = await _phase_verification(topic, report)

    # Save results
    json_path, md_path = _save_results(topic, report, validation, output_dir)

    logger.info(
        "research_pipeline_complete",
        json_output=str(json_path),
        md_output=str(md_path),
        confidence=validation.confidence_score,
    )

    return report, validation


def main() -> None:
    """CLI entry point for deep research."""
    parser = argparse.ArgumentParser(
        description="Run deep research on a topic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.agents.pydantic.research --topic "PydanticAI features"
    python -m src.agents.pydantic.research --topic "LangGraph patterns" --output ./my_research
        """,
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="The research topic to investigate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None

    print(f"\n{'=' * 60}")
    print(f"Deep Research: {args.topic}")
    print(f"{'=' * 60}\n")

    report, validation = asyncio.run(run_research(args.topic, output_dir))

    print(f"\n{'=' * 60}")
    print("Research Complete!")
    print(f"{'=' * 60}")
    print(f"\nTitle: {report.title}")
    print(f"\nSummary:\n{report.summary}")
    print("\nKey Findings:")
    for finding in report.key_findings:
        print(f"  - {finding}")
    print(f"\nConfidence Score: {validation.confidence_score:.2f}")
    print(f"Valid: {'Yes' if validation.is_valid else 'No'}")
    if validation.issues_found:
        print("\nIssues Found:")
        for issue in validation.issues_found:
            print(f"  - {issue}")
    print(f"\nResults saved to: {output_dir or DEFAULT_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
