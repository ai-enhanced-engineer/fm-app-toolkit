"""Text analysis agent with tools and dependency injection using PydanticAI.

This module demonstrates:
- Tool registration and usage with agents
- Dependency injection with runtime context
- Context-aware tool functions
"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.tools import word_count


class AnalysisResult(BaseModel):
    """Structured output for text analysis with tool usage."""

    original_text: str = Field(description="The input text")
    word_count: int = Field(description="Number of words")
    sentiment: str = Field(description="Detected sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")
    key_insights: list[str] = Field(default_factory=list, description="Key insights from analysis")


# Dependency injection example
@dataclass
class AnalysisContext:
    """Runtime context for analysis agents."""

    user_id: str
    session_id: str
    config: dict[str, Any]
    debug: bool = False


def create_analysis_agent(model: str) -> Agent[AnalysisContext, AnalysisResult]:
    """Create an agent that analyzes text using tools and context."""
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
        return word_count(text)

    @agent.tool
    def calculate_confidence(ctx: RunContext[AnalysisContext], sentiment: str) -> float:
        """Calculate confidence score based on analysis context."""
        base_confidence = 0.7
        high_precision = ctx.deps.config.get("high_precision", False)

        # Adjust confidence based on context
        if high_precision:
            base_confidence += 0.2

        if sentiment in ["positive", "negative"]:
            final_confidence = min(base_confidence + 0.1, 1.0)
            return final_confidence

        return base_confidence

    return agent


async def example_usage(agent: Agent[AnalysisContext, AnalysisResult], text: str) -> None:
    """Demonstrate the analysis agent with sample input."""
    # Create sample context
    context = AnalysisContext(
        user_id="demo_user",
        session_id="demo_session",
        config={"high_precision": True},
        debug=True,
    )

    print("=== Text Analysis Agent Demo ===")
    print(f"Input: {text}")
    print(f"Context: User={context.user_id}, High Precision={context.config.get('high_precision')}")

    # Run analysis
    result = await agent.run(text, deps=context)

    print(f"Sentiment: {result.output.sentiment}")
    print(f"Confidence: {result.output.confidence}")
    print(f"Word count: {result.output.word_count}")
    print(f"Key insights: {result.output.key_insights}")


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
    args = parser.parse_args()

    print(f"Creating analysis agent with model: {args.model}")

    # Create agent with specified model
    agent = create_analysis_agent(args.model)

    # Run example with specified text
    asyncio.run(example_usage(agent, args.text))
