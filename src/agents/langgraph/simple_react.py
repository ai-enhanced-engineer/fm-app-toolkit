"""Simple ReAct agent using LangGraph's create_react_agent prebuilt.

This module demonstrates production-ready ReAct pattern using LangGraph's
prebuilt create_react_agent function. It provides a clean API while handling
all the complexity of state management and tool routing internally.

ReAct Pattern (Yao et al., 2022): https://arxiv.org/abs/2210.03629

Key features:
- Uses create_react_agent for minimal boilerplate
- Automatic tool routing and state management
- Built-in error handling and observability
- Clean API matching LlamaIndex agent interface

For learning how ReAct works internally, see minimal_react.py which shows
the explicit graph construction.
"""

from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool

# Note: create_react_agent deprecated in LangGraph v1.0, will be removed in v2.0
# Future migration: Use `from langchain.agents import create_react_agent` instead
# See: https://github.com/langchain-ai/langgraph/issues/6404
from langgraph.prebuilt import create_react_agent  # pyright: ignore[reportDeprecated]


@dataclass
class Tool:
    """Simple tool representation - a function with metadata."""

    name: str
    description: str
    function: Callable[..., Any]


class SimpleReActAgent:
    """A simple ReAct agent using create_react_agent prebuilt.

    This provides a production-ready ReAct agent with minimal code.
    The create_react_agent function handles:
    - Graph structure and state management
    - Tool routing and execution
    - Error handling and retries
    - Message formatting

    Example:
        >>> from src.agents.langgraph.simple_react import SimpleReActAgent, Tool
        >>> from langchain_openai import ChatOpenAI
        >>> tools = [
        ...     Tool("calculate", "Do math", calculate),
        ...     Tool("get_weather", "Get weather for a city", get_weather)
        ... ]
        >>> agent = SimpleReActAgent(llm=ChatOpenAI(), tools=tools, verbose=True)
        >>> result = await agent.run("What's the weather in Tokyo?")
    """

    def __init__(self, llm: BaseChatModel, tools: list[Tool], verbose: bool = False) -> None:
        """Initialize the agent with an LLM and tools.

        Args:
            llm: LangChain chat model to use for reasoning
            tools: List of tools the agent can use
            verbose: If True, print step-by-step execution for transparency
        """
        self.llm = llm
        self.verbose = verbose

        # Convert our Tool objects to LangChain tools
        self.langchain_tools = []
        for t in tools:
            # Create a LangChain tool from our function
            # The @tool decorator returns a function that creates the tool
            lc_tool = langchain_tool(t.function)
            lc_tool.name = t.name
            lc_tool.description = t.description
            self.langchain_tools.append(lc_tool)

        # Create the ReAct agent using the prebuilt function
        # This handles all the graph construction, state management, and routing
        self.agent = create_react_agent(model=llm, tools=self.langchain_tools)  # pyright: ignore[reportDeprecated]

    async def run(self, query: str) -> dict[str, Any]:
        """Run the agent on a query.

        The create_react_agent handles the ReAct loop internally:
        1. Receives query
        2. Calls LLM to reason and select tools
        3. Executes tools and adds observations
        4. Loops until final answer
        5. Returns result

        Args:
            query: User's question or request

        Returns:
            Dictionary with 'response', 'reasoning' (messages), 'sources' (tool outputs)
        """
        if self.verbose:
            print("\n🚀 Starting ReAct Agent")
            print("=" * 60)
            print(f"Query: {query}")

        # Invoke the agent with the user's query
        # The agent graph handles all the state transitions
        try:
            result = await self.agent.ainvoke({"messages": [HumanMessage(content=query)]})
        except Exception as e:
            return {
                "response": f"Error during agent execution: {str(e)}",
                "reasoning": [],
                "sources": [],
            }

        # Extract messages from result
        messages = result.get("messages", [])

        # Track reasoning steps
        step_count = 0

        # Get the final response (last AI message content)
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        final_response = ai_messages[-1].content if ai_messages else "No response generated"

        # Extract tool outputs as sources
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        sources = [m.content for m in tool_messages]

        # Format reasoning steps (all AI messages)
        reasoning = []
        for m in ai_messages:
            step_count += 1
            reasoning.append(m.content or "")

            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Step {step_count}: Agent reasoning")
                print("=" * 60)

                # Check if this message has tool calls
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        print(f"🔧 Tool call: {tc['name']}")
                        print(f"📝 Arguments: {tc['args']}")
                else:
                    content_preview = (m.content or "")[:100]
                    print(f"💬 Response: {content_preview}...")

        # Print tool observations if verbose
        if self.verbose and sources:
            print(f"\n{'=' * 60}")
            print("Tool observations:")
            print("=" * 60)
            for i, source in enumerate(sources, 1):
                source_preview = str(source)[:100]
                print(f"👀 Observation {i}: {source_preview}...")

        if self.verbose:
            print(f"\n{'=' * 60}")
            print("✅ Agent complete")
            print("=" * 60)
            print(f"Response: {final_response}")
            print(f"Reasoning steps: {len(reasoning)}")
            print(f"Sources: {len(sources)}\n")

        return {"response": final_response, "reasoning": reasoning, "sources": sources}
