"""Minimal educational ReAct agent using LangGraph for learning purposes.

This module demonstrates the ReAct (Reasoning + Acting) pattern using LangGraph's
StateGraph for explicit control flow. It's designed for pedagogy, not production use.

ReAct Pattern (Yao et al., 2022): https://arxiv.org/abs/2210.03629

Key differences from simple_react.py:
- Explicit graph construction - you can see every node and edge
- Custom state management - no hidden abstractions
- Manual tool routing - the conditional logic is visible
- Heavily commented for learning

For production use, see simple_react.py which uses create_react_agent prebuilt
and provides better error handling and observability.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.common import Tool


# Define the agent's state - this is what gets passed between graph nodes
class AgentState(TypedDict):
    """State for the ReAct agent graph.

    The messages list accumulates as the agent reasons and acts.
    The add_messages reducer ensures messages are appended, not replaced.
    """

    messages: Annotated[list[BaseMessage], operator.add]


class MinimalReActAgent:
    """A minimal ReAct agent with explicit graph construction for learning.

    This implementation makes the LangGraph state machine visible:
    1. Start with user's query as HumanMessage
    2. Call model to get reasoning and optional tool call
    3. If tool call present, execute tool and add ToolMessage
    4. Loop back to model with observation
    5. When no tool calls, return final answer

    The graph structure is:
        START → agent → should_continue? → [tools → agent] OR [END]

    Example:
        >>> from src.agents.langgraph.minimal_react import MinimalReActAgent, Tool
        >>> from langchain_openai import ChatOpenAI
        >>> tools = [
        ...     Tool("calculate", "Do math", calculate),
        ...     Tool("get_weather", "Get weather for a city", get_weather)
        ... ]
        >>> agent = MinimalReActAgent(llm=ChatOpenAI(), tools=tools, max_steps=5)
        >>> result = await agent.run("What's the weather in Tokyo?")
    """

    def __init__(self, llm: BaseChatModel, tools: list[Tool], max_steps: int = 10, verbose: bool = False) -> None:
        """Initialize the agent with an LLM and tools.

        Args:
            llm: LangChain chat model to use for reasoning
            tools: List of tools the agent can use
            max_steps: Maximum reasoning iterations before giving up
            verbose: If True, print step-by-step execution for transparency
        """
        self.llm = llm
        self.max_steps = max_steps
        self.verbose = verbose

        # Convert our Tool objects to LangChain tools for binding
        self.langchain_tools = []
        for t in tools:
            # Create a LangChain tool from our function
            # The @tool decorator returns a function that creates the tool
            lc_tool = langchain_tool(t.function)
            lc_tool.name = t.name
            lc_tool.description = t.description
            self.langchain_tools.append(lc_tool)

        # Bind tools to the LLM so it can call them
        self.llm_with_tools = self.llm.bind_tools(self.langchain_tools)

        # Build the graph - this is where the ReAct loop structure is defined
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:  # type: ignore[type-arg]
        """Build the LangGraph state machine for ReAct.

        The graph has 2 nodes:
        - agent: Calls the LLM to reason and potentially select a tool
        - tools: Executes the selected tool and adds observation

        The conditional edge "should_continue" routes based on whether
        the LLM called a tool or provided a final answer.
        """
        # Create the graph with our state schema
        workflow = StateGraph(AgentState)

        # Add nodes - these are the functions that process state
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._execute_tools)

        # Set the entry point - start with the agent node
        workflow.set_entry_point("agent")

        # Add conditional edges - this is the routing logic
        # After agent runs, check if we should continue or end
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",  # If tools called, go to tools node
                "end": END,  # If done, end the graph
            },
        )

        # After tools execute, always go back to agent
        workflow.add_edge("tools", "agent")

        # Compile the graph into a runnable
        return workflow.compile()

    def _call_model(self, state: AgentState) -> dict[str, list[BaseMessage]]:
        """Node function: Call the LLM with current messages.

        This is where the LLM does its reasoning and decides whether
        to use a tool or provide a final answer.

        Args:
            state: Current agent state with message history

        Returns:
            Dictionary with new messages to add to state
        """
        messages = state["messages"]

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"🤖 Agent reasoning step {len([m for m in messages if isinstance(m, AIMessage)]) + 1}")
            print(f"{'=' * 60}")

        # Call the LLM with the full message history
        response = self.llm_with_tools.invoke(messages)

        if self.verbose:
            if hasattr(response, "tool_calls") and response.tool_calls:
                print(f"🔧 Tool call requested: {response.tool_calls[0]['name']}")
            else:
                content_preview = (response.content or "")[:100]
                print(f"💬 Response: {content_preview}...")

        # Return new messages to add to state
        return {"messages": [response]}

    def _execute_tools(self, state: AgentState) -> dict[str, list[BaseMessage]]:
        """Node function: Execute tools requested by the LLM.

        Extracts tool calls from the last AI message, executes them,
        and creates ToolMessage observations to add to the state.

        Args:
            state: Current agent state with message history

        Returns:
            Dictionary with tool result messages to add to state
        """
        messages = state["messages"]
        last_message = messages[-1]

        # Extract tool calls from the AI message
        tool_calls = getattr(last_message, "tool_calls", [])

        if self.verbose:
            print(f"🛠️  Executing {len(tool_calls)} tool(s)...")

        # Execute each tool and collect results
        tool_messages: list[BaseMessage] = []
        for tool_call in tool_calls:
            # Find the matching LangChain tool
            tool = next((t for t in self.langchain_tools if t.name == tool_call["name"]), None)

            if tool is None:
                result = f"Error: Tool '{tool_call['name']}' not found"
            else:
                try:
                    # Execute the tool with provided arguments
                    result = tool.invoke(tool_call["args"])
                except Exception as e:
                    result = f"Error executing {tool_call['name']}: {str(e)}"

            if self.verbose:
                result_preview = str(result)[:100]
                print(f"👀 Observation: {result_preview}...")

            # Create a ToolMessage with the result
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

        return {"messages": tool_messages}

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Conditional edge: Determine if we should continue or end.

        Checks if the last message has tool calls. If yes, we need to
        execute tools and continue reasoning. If no, we're done.

        Args:
            state: Current agent state with message history

        Returns:
            "continue" if tools need execution, "end" if reasoning is complete
        """
        messages = state["messages"]
        last_message = messages[-1]

        # Check if the LLM called any tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    async def run(self, query: str) -> dict[str, Any]:
        """Run the agent on a query using the LangGraph state machine.

        This is the entry point that kicks off the ReAct loop.

        Args:
            query: User's question or request

        Returns:
            Dictionary with 'response', 'reasoning' (messages), 'sources' (tool outputs)
        """
        if self.verbose:
            print("\n🚀 Starting ReAct Agent")
            print("=" * 60)
            print(f"Query: {query}")

        # Initialize state with the user's query
        initial_state: AgentState = {"messages": [HumanMessage(content=query)]}

        # Run the graph - it will loop through agent → tools → agent until done
        try:
            final_state = await self.graph.ainvoke(initial_state)
        except Exception as e:
            return {
                "response": f"Error during agent execution: {str(e)}",
                "reasoning": [],
                "sources": [],
            }

        # Extract results from final state
        messages: list[BaseMessage] = final_state["messages"]

        # Get the final response (last AI message)
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        final_response = ai_messages[-1].content if ai_messages else "No response generated"

        # Extract tool outputs as sources
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        sources = [m.content for m in tool_messages]

        # Format reasoning steps (all AI messages)
        reasoning = [m.content for m in ai_messages]

        if self.verbose:
            print(f"\n{'=' * 60}")
            print("✅ Agent complete")
            print("=" * 60)
            print(f"Response: {final_response}")
            print(f"Reasoning steps: {len(reasoning)}")
            print(f"Sources: {len(sources)}\n")

        return {"response": final_response, "reasoning": reasoning, "sources": sources}
