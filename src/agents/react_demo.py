"""ReAct agent demo demonstrating tool-based reasoning.

Run with: make react-demo
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from llama_index.core.agent.workflow import ReActAgent  # noqa: E402
from llama_index.core.tools import FunctionTool  # noqa: E402
from llama_index.llms.openai import OpenAI  # noqa: E402


# Mock tools that return data relevant to enterprise discount inquiry
def search_kb() -> dict[str, object]:
    """Search the knowledge base for relevant information."""
    return {
        "result": "Found enterprise pricing documentation",
        "articles": ["Enterprise Pricing Guide", "Volume Discounts", "Custom Plans"],
    }


def get_pricing() -> dict[str, object]:
    """Get current pricing information."""
    return {
        "plan": "Enterprise",
        "base_price": 299.99,
        "currency": "USD",
        "billing": "annual",
        "includes": ["Priority support", "Custom integrations", "SLA guarantee"],
    }


def calculate_discount() -> dict[str, object]:
    """Calculate applicable discount for the customer."""
    return {
        "discount_percent": 25,
        "reason": "Enterprise volume discount",
        "final_price": 224.99,
        "savings": 75.00,
    }


async def run_react_agent(query: str) -> str:
    """Run a ReAct agent with the given query."""
    llm = OpenAI(model="gpt-4.1-mini")

    # Create tools
    search_tool = FunctionTool.from_defaults(fn=search_kb, name="search_kb")
    pricing_tool = FunctionTool.from_defaults(fn=get_pricing, name="get_pricing")
    discount_tool = FunctionTool.from_defaults(fn=calculate_discount, name="calculate_discount")

    # Create agent with tools and LLM
    agent = ReActAgent(tools=[search_tool, pricing_tool, discount_tool], llm=llm)

    # Run the agent
    response = await agent.run(user_msg=query)
    return str(response)


def main() -> None:
    """Run the ReAct agent example."""
    query = "What's the discount for enterprise customers?"

    print(f"🤖 Query: {query}")
    print("-" * 50)

    response = asyncio.run(run_react_agent(query))

    print("-" * 50)
    print(f"📝 Response: {response}")


if __name__ == "__main__":
    main()
