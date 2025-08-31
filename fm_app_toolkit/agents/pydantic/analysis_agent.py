"""Text analysis agent with tools and dependency injection using PydanticAI.

This module demonstrates:
- Tool registration and usage with agents
- Dependency injection with runtime context
- Context-aware tool functions
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from fm_app_toolkit.tools import word_count

# Load environment variables (for OpenAI API key, etc.)
load_dotenv()

# Configure Logfire for observability and monitoring (only if authenticated)
try:
    logfire.configure()
    logfire.instrument_pydantic_ai()
    _LOGFIRE_ENABLED = True
except Exception:
    # Logfire not configured or authenticated, disable features gracefully
    _LOGFIRE_ENABLED = False


def _safe_logfire_span(name: str, **kwargs) -> Any:
    """Create a logfire span if enabled, otherwise a no-op context manager."""
    if _LOGFIRE_ENABLED:
        return logfire.span(name, **kwargs)
    else:
        from contextlib import nullcontext

        return nullcontext()


def _safe_logfire_info(message: str, **kwargs) -> None:
    """Log info with logfire if enabled, otherwise do nothing."""
    if _LOGFIRE_ENABLED:
        logfire.info(message, **kwargs)


class AnalysisResult(BaseModel):
    """Structured output for text analysis with tool usage."""

    original_text: str = Field(description="The input text")
    word_count: int = Field(description="Number of words")
    sentiment: str = Field(description="Detected sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")
    key_insights: List[str] = Field(default_factory=list, description="Key insights from analysis")


# Dependency injection example
@dataclass
class AnalysisContext:
    """Runtime context for analysis agents."""

    user_id: str
    session_id: str
    config: Dict[str, Any]
    debug: bool = False


def create_analysis_agent(model: str) -> Agent[AnalysisContext, AnalysisResult]:
    """Create an agent that analyzes text using tools and context.

    This demonstrates:
    - Tool registration and usage
    - Dependency injection with context
    - Complex structured outputs

    Args:
        model: Model string specification (e.g., 'openai:gpt-4o').

    Returns:
        Agent configured for text analysis with tools.
    """
    agent = Agent(
        model,
        deps_type=AnalysisContext,
        output_type=AnalysisResult,
        system_prompt="""You are a text analysis expert.
        Analyze the provided text for sentiment and key insights.
        Use available tools to enhance your analysis.""",
        instrument=True,  # Enable Logfire instrumentation for observability
    )

    # Register tools with the agent
    @agent.tool
    def count_words_with_context(ctx: RunContext[AnalysisContext], text: str) -> int:
        """Count words with access to context."""
        with _safe_logfire_span("word_count_tool", user_id=ctx.deps.user_id, session_id=ctx.deps.session_id):
            _safe_logfire_info(
                "Counting words for analysis",
                user_id=ctx.deps.user_id,
                text_length=len(text),
                debug_mode=ctx.deps.debug,
            )
            result = word_count(text)
            _safe_logfire_info("Word count completed", word_count=result)
            return result

    @agent.tool
    def calculate_confidence(ctx: RunContext[AnalysisContext], sentiment: str) -> float:
        """Calculate confidence score based on analysis context."""
        with _safe_logfire_span("confidence_calculation_tool", user_id=ctx.deps.user_id, sentiment=sentiment):
            base_confidence = 0.7
            high_precision = ctx.deps.config.get("high_precision", False)

            _safe_logfire_info(
                "Calculating confidence score",
                sentiment=sentiment,
                high_precision=high_precision,
                base_confidence=base_confidence,
            )

            # Adjust confidence based on context
            if high_precision:
                base_confidence += 0.2
                _safe_logfire_info("Applied high precision bonus", adjusted_confidence=base_confidence)

            if sentiment in ["positive", "negative"]:
                final_confidence = min(base_confidence + 0.1, 1.0)
                _safe_logfire_info("Applied sentiment bonus", final_confidence=final_confidence)
                return final_confidence

            _safe_logfire_info("Confidence calculation completed", final_confidence=base_confidence)
            return base_confidence

    return agent


async def example_usage(agent: Agent[AnalysisContext, AnalysisResult], text: str) -> None:
    """Demonstrate the analysis agent with sample input."""
    with _safe_logfire_span("analysis_agent_demo") as span:
        # Create sample context
        context = AnalysisContext(
            user_id="demo_user",
            session_id="demo_session",
            config={"high_precision": True},
            debug=True,
        )

        _safe_logfire_info(
            "Starting text analysis demo",
            input_text=text[:100] + "..." if len(text) > 100 else text,
            user_id=context.user_id,
            session_id=context.session_id,
            high_precision=context.config.get("high_precision"),
            debug_mode=context.debug,
        )

        print("=== Text Analysis Agent Demo ===")
        print(f"Input: {text}")
        print(f"Context: User={context.user_id}, High Precision={context.config.get('high_precision')}")

        # Run analysis with timing
        _safe_logfire_info("Executing agent analysis")
        result = await agent.run(text, deps=context)

        # Log results
        _safe_logfire_info(
            "Analysis completed successfully",
            sentiment=result.output.sentiment,
            confidence=result.output.confidence,
            word_count=result.output.word_count,
            insights_count=len(result.output.key_insights),
        )

        print(f"Sentiment: {result.output.sentiment}")
        print(f"Confidence: {result.output.confidence}")
        print(f"Word count: {result.output.word_count}")
        print(f"Key insights: {result.output.key_insights}")

        if _LOGFIRE_ENABLED and hasattr(span, "set_attribute"):
            span.set_attribute("analysis.sentiment", result.output.sentiment)
            span.set_attribute("analysis.confidence", result.output.confidence)
            span.set_attribute("analysis.word_count", result.output.word_count)


if __name__ == "__main__":
    import argparse
    import asyncio

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run PydanticAI analysis agent")
    parser.add_argument("--model", type=str, required=True, help="Model specification (e.g., 'openai:gpt-4o')")
    parser.add_argument(
        "--text",
        type=str,
        default="This product exceeded all my expectations! Absolutely fantastic quality and service.",
        help="Text to analyze (default: positive product review)",
    )
    parser.add_argument(
        "--enable-logfire", action="store_true", default=True, help="Enable Logfire observability (default: enabled)"
    )
    parser.add_argument("--disable-logfire", action="store_true", help="Disable Logfire observability")

    args = parser.parse_args()

    # Handle Logfire configuration based on CLI flags
    enable_logfire = args.enable_logfire and not args.disable_logfire

    if not enable_logfire and _LOGFIRE_ENABLED:
        # Disable Logfire if requested and it's available
        try:
            logfire.configure(send_to_logfire=False)
            print("Logfire observability disabled")
        except Exception:
            print("Logfire observability not available")
    elif _LOGFIRE_ENABLED:
        print("Logfire observability enabled")
    else:
        print("Logfire observability not available (not authenticated)")

    print(f"Creating analysis agent with model: {args.model}")

    with _safe_logfire_span("main_execution", model=args.model, logfire_enabled=enable_logfire):
        # Create agent with specified model
        agent = create_analysis_agent(args.model)

        # Run example with specified text
        asyncio.run(example_usage(agent, args.text))
