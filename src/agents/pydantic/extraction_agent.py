"""Data extraction agent using PydanticAI.

This module demonstrates:
- Extracting structured data from unstructured text
- Working with complex nested Pydantic models
- List fields with default factories
"""

import argparse
import asyncio

from pydantic import BaseModel, Field
from pydantic_ai import Agent


class DataExtraction(BaseModel):
    """Structured output for extracting data from text."""

    entities: list[str] = Field(default_factory=list, description="Named entities found")
    numbers: list[float] = Field(default_factory=list, description="Numbers mentioned")
    key_phrases: list[str] = Field(default_factory=list, description="Important phrases")
    summary: str = Field(description="Brief summary of the text")
    word_count: int = Field(description="Total word count")


def create_extraction_agent(model: str) -> Agent[None, DataExtraction]:
    """Create an agent that extracts structured data from unstructured text."""
    return Agent(
        model,
        output_type=DataExtraction,
        system_prompt="""Extract structured information from the provided text.
        Identify entities, numbers, key phrases, and provide a summary.
        Be thorough but concise.""",
    )


async def example_usage(agent: Agent[None, DataExtraction], text1: str, text2: str) -> None:
    """Demonstrate extraction agent with various text types."""

    # Example 1: Business text with entities and numbers
    result1 = await agent.run(text1)

    print("=== Data Extraction Example 1 ===")
    print(f"Input: {text1}")
    print(f"Entities: {result1.output.entities}")
    print(f"Numbers: {result1.output.numbers}")
    print(f"Key phrases: {result1.output.key_phrases}")
    print(f"Summary: {result1.output.summary}")
    print(f"Word count: {result1.output.word_count}")

    print("\n" + "=" * 50 + "\n")

    # Example 2: Technical text
    result2 = await agent.run(text2)

    print("=== Data Extraction Example 2 ===")
    print(f"Input: {text2}")
    print(f"Numbers found: {result2.output.numbers}")
    print(f"Key technical terms: {result2.output.key_phrases}")
    print(f"Summary: {result2.output.summary}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run PydanticAI extraction agent")
    parser.add_argument("--model", type=str, required=True, help="Model specification (e.g., 'openai:gpt-4o')")
    parser.add_argument(
        "--text1",
        type=str,
        default="Apple reported $394.3 billion in revenue for 2022, with iPhone sales leading growth.",
        help="First text to analyze (default: business text)",
    )
    parser.add_argument(
        "--text2",
        type=str,
        default="The new processor runs at 3.5GHz with 8 cores and 16 threads, consuming only 65 watts.",
        help="Second text to analyze (default: technical text)",
    )

    args = parser.parse_args()

    print(f"Creating extraction agent with model: {args.model}")

    # Create agent with specified model
    agent = create_extraction_agent(args.model)

    # Run example with specified texts
    asyncio.run(example_usage(agent, args.text1, args.text2))
