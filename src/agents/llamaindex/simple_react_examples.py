"""Production examples demonstrating SimpleReActAgent usage patterns.

This module provides runnable examples showing how to use the SimpleReActAgent
in production scenarios. It demonstrates:
- Tool setup and configuration
- Agent initialization with various parameters
- Different query patterns (calculation, time, weather, search)
- Result extraction and display

Run directly:
    python -m src.agents.llamaindex.simple_react_examples --model openai:gpt-4

Or import the example function:
    from src.agents.llamaindex.simple_react_examples import example_usage
    await example_usage(llm)
"""

from llama_index.core.llms.llm import LLM

from src.agents.llamaindex.simple_react import SimpleReActAgent, Tool


async def example_usage(llm: LLM) -> None:
    """Demonstrate the SimpleReActAgent with sample tools.

    Args:
        llm: The language model to use for the agent
    """
    from src.agents.llamaindex.sample_tools import (
        calculate,
        get_current_time,
        get_weather,
        search_web,
    )

    # Create sample tools
    tools = [
        Tool(
            name="get_current_time",
            function=get_current_time,
            description="Get the current date and time in UTC",
        ),
        Tool(
            name="calculate",
            function=calculate,
            description="Perform mathematical calculations. Input should be a mathematical expression.",
        ),
        Tool(
            name="get_weather",
            function=get_weather,
            description="Get the current weather for a city. Input should be the city name.",
        ),
        Tool(
            name="search_web",
            function=search_web,
            description="Search the web for information. Input should be a search query.",
        ),
    ]

    # Create the agent
    agent = SimpleReActAgent(
        llm=llm,
        system_header="You are a helpful assistant with access to various tools.",
        extra_context="Always think step by step and use tools when needed to provide accurate information.",
        max_reasoning=10,
        tools=tools,
        verbose=True,
    )

    print("=" * 60)
    print("🤖 LlamaIndex ReAct Agent Demo")
    print("=" * 60)
    print()

    # Example 1: Simple calculation
    print("📝 Example 1: Mathematical Calculation")
    print("-" * 40)
    query1 = "What is 15 times 7 plus 23?"
    print(f"Query: {query1}")
    print()

    handler1 = agent.run(user_msg=query1)
    result1 = await agent.get_results_from_handler(handler1)

    print(f"✅ Response: {result1['response']}")
    print(f"🔧 Tools Used: {result1['sources']}")
    print()

    # Example 2: Current time
    print("📝 Example 2: Current Time")
    print("-" * 40)
    query2 = "What's the current time?"
    print(f"Query: {query2}")
    print()

    handler2 = agent.run(user_msg=query2)
    result2 = await agent.get_results_from_handler(handler2)

    print(f"✅ Response: {result2['response']}")
    print()

    # Example 3: Multi-step reasoning with weather
    print("📝 Example 3: Weather Information")
    print("-" * 40)
    query3 = "What's the weather like in Tokyo and New York? Compare them."
    print(f"Query: {query3}")
    print()

    handler3 = agent.run(user_msg=query3)
    result3 = await agent.get_results_from_handler(handler3)

    print(f"✅ Response: {result3['response']}")
    print(f"🔧 Tools Used: {len(result3['sources'])} tool calls")
    print()

    # Example 4: Web search
    print("📝 Example 4: Web Search")
    print("-" * 40)
    query4 = "Search for information about the latest developments in quantum computing."
    print(f"Query: {query4}")
    print()

    handler4 = agent.run(user_msg=query4)
    result4 = await agent.get_results_from_handler(handler4)

    print(f"✅ Response: {result4['response']}")
    print()

    print("=" * 60)
    print("✨ Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    import asyncio

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LlamaIndex SimpleReActAgent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model specification (e.g., 'openai:gpt-4', 'anthropic:claude-3')",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional custom query to run instead of examples",
    )

    args = parser.parse_args()

    # Parse model string to determine provider
    model_parts = args.model.split(":")
    if len(model_parts) != 2:
        print("Error: Model should be in format 'provider:model' (e.g., 'openai:gpt-4')")
        exit(1)

    provider, model_name = model_parts

    # Create OpenAI LLM
    if provider.lower() == "openai":
        from llama_index.llms.openai import OpenAI

        llm = OpenAI(model=model_name)
    else:
        print(f"Error: Unsupported provider '{provider}'. Currently only 'openai' is supported.")
        exit(1)

    print(f"🚀 Initializing SimpleReActAgent with {args.model}...")
    print()

    # Run custom query or examples
    if args.query:
        # Custom query mode
        from src.agents.llamaindex.sample_tools import (
            calculate,
            get_current_time,
            get_weather,
            search_web,
        )

        async def run_custom_query() -> None:
            tools = [
                Tool(
                    name="get_current_time",
                    function=get_current_time,
                    description="Get the current date and time in UTC",
                ),
                Tool(
                    name="calculate",
                    function=calculate,
                    description="Perform mathematical calculations",
                ),
                Tool(
                    name="get_weather",
                    function=get_weather,
                    description="Get weather for a city",
                ),
                Tool(
                    name="search_web",
                    function=search_web,
                    description="Search the web for information",
                ),
            ]

            agent = SimpleReActAgent(
                llm=llm,
                system_header="You are a helpful assistant with access to various tools.",
                tools=tools,
                verbose=True,
            )

            print(f"Query: {args.query}")
            print()

            handler = agent.run(user_msg=args.query)
            result = await agent.get_results_from_handler(handler)

            print(f"Response: {result['response']}")
            if result["sources"]:
                print(f"Sources: {result['sources']}")

        asyncio.run(run_custom_query())
    else:
        # Run example usage
        asyncio.run(example_usage(llm))
