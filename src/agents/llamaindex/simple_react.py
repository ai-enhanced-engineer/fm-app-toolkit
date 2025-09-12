"""A minimalistic ReAct agent implementation using BaseWorkflowAgent.

This module demonstrates the core ReAct (Reasoning + Acting) pattern using
llama_index's BaseWorkflowAgent architecture:
1. Thought: The agent reasons about what to do
2. Action: The agent decides to use a tool (optional)
3. Observation: The agent observes the tool's output
4. Answer: The agent provides a final response

This is a custom implementation that extends BaseWorkflowAgent while maintaining
the pedagogical clarity of the ReAct pattern.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union, cast

from llama_index.core.agent.react import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput,
    ToolCallResult,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory
from llama_index.core.tools import (
    AsyncBaseTool,
    FunctionTool,
    ToolSelection,
)
from llama_index.core.workflow import Context

from src.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Context key for storing current reasoning steps
CTX_CURRENT_REASONING = "current_reasoning"
CTX_SOURCES = "sources"


@dataclass
class Tool:
    """Simple tool representation."""

    name: str
    function: Callable[..., Any]
    description: str


class SimpleReActAgent(BaseWorkflowAgent):
    """A minimalistic ReAct agent using BaseWorkflowAgent.

    This agent follows the ReAct pattern by implementing the required
    abstract methods:
    - take_step: Get LLM response and parse it
    - handle_tool_call_results: Process tool outputs
    - finalize: Clean up and format final response

    This custom implementation demonstrates how to build a ReAct agent
    while leveraging LlamaIndex's BaseWorkflowAgent infrastructure.
    """

    def __init__(
        self,
        llm: LLM,
        system_header: str,
        extra_context: Optional[str] = None,
        max_reasoning: Optional[int] = None,
        tools: Optional[List[Tool]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        # Convert our Tool objects to FunctionTools for BaseWorkflowAgent
        function_tools: List[Union[FunctionTool, Callable[..., Any]]] = []
        if tools:
            for tool in tools:
                try:
                    func_tool = FunctionTool.from_defaults(
                        fn=tool.function, name=tool.name, description=tool.description
                    )
                    function_tools.append(func_tool)
                except Exception as e:
                    logger.warning("Failed to register tool", tool_name=tool.name, error=str(e))
                    if self._verbose:
                        print(f"Warning: Failed to register tool '{tool.name}': {e}")
                    # Continue processing other tools

        # Combine system header and extra context for system prompt
        system_prompt = system_header
        if extra_context:
            system_prompt = f"{system_header}\n\n{extra_context}"

        # Initialize BaseWorkflowAgent
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            tools=function_tools,
            name="SimpleReActAgent",
            description="A pedagogical ReAct agent implementation",
            **kwargs,
        )

        # Store additional configuration
        self._max_reasoning = max_reasoning if max_reasoning is not None else 15
        self._verbose = verbose
        self._system_header = system_header
        self._extra_context = extra_context if extra_context is not None else ""

        # Initialize logger for this instance
        self._logger = get_logger(f"{__name__}.{self.name}")

        # Initialize formatter with system header and extra context
        self._formatter = ReActChatFormatter(system_header=self._system_header, context=self._extra_context)

        # Initialize parser - using LlamaIndex's built-in ReActOutputParser
        self._output_parser = ReActOutputParser()

    async def take_step(
        self,
        ctx: Context,
        llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
        memory: BaseMemory,
    ) -> AgentOutput:
        """Take a single step with the ReAct agent.

        This implements the core ReAct reasoning loop:
        1. Format input with current reasoning steps
        2. Get LLM response
        3. Parse the response to determine action or answer
        4. Return appropriate AgentOutput
        """
        # Get or initialize current reasoning from context
        current_reasoning: List[BaseReasoningStep] = await ctx.store.get(CTX_CURRENT_REASONING, default=[])

        # Get or initialize sources from context
        # Not used in this method but needed in context
        await ctx.store.get(CTX_SOURCES, default=[])

        # Check if we've exceeded max reasoning steps
        if len(current_reasoning) >= self._max_reasoning:
            if self._verbose:
                self._logger.info(
                    "Exceeded max reasoning steps",
                    max_reasoning=self._max_reasoning,
                    current_steps=len(current_reasoning),
                )

            # Add final response step
            current_reasoning.append(
                ResponseReasoningStep(
                    thought="Exceeded max reasoning steps",
                    response="I couldn't complete the reasoning in the allowed iterations.",
                    is_streaming=False,
                )
            )
            await ctx.store.set(CTX_CURRENT_REASONING, current_reasoning)

            return AgentOutput(
                response=ChatMessage(
                    role="assistant", content="I couldn't complete the reasoning in the allowed iterations."
                ),
                current_agent_name=self.name,
                raw=None,
            )

        # Remove system prompt if present, as BaseWorkflowAgent already handles it
        if llm_input and llm_input[0].role == "system":
            llm_input = llm_input[1:]

        # Format input using ReActChatFormatter
        formatted_messages = self._formatter.format(
            tools, llm_input, current_reasoning=current_reasoning if current_reasoning else None
        )

        if self._verbose:
            self._logger.debug("Starting reasoning step", step_number=len(current_reasoning) + 1)

        # Get LLM response
        response = await self.llm.achat(formatted_messages)
        llm_output = response.message.content

        if self._verbose:
            self._logger.debug(
                "LLM response received",
                output_length=len(llm_output) if llm_output else 0,
                output_preview=llm_output[:100] if llm_output else None,
            )

        # Parse the output
        try:
            reasoning_step = self._output_parser.parse(llm_output or "", is_streaming=False)
        except ValueError as e:
            if self._verbose:
                self._logger.warning(
                    "Failed to parse LLM output", error=str(e), output=llm_output[:200] if llm_output else None
                )

            # Return error with retry messages
            return AgentOutput(
                response=response.message,
                current_agent_name=self.name,
                raw=response.raw if hasattr(response, "raw") else None,
                retry_messages=[
                    response.message,
                    ChatMessage(
                        role="user", content=f"Error parsing output: {e}\\n\\nPlease format your response correctly."
                    ),
                ],
            )

        # Add reasoning step to context
        current_reasoning.append(reasoning_step)
        await ctx.store.set(CTX_CURRENT_REASONING, current_reasoning)

        # Handle different reasoning step types
        if reasoning_step.is_done:
            # Response step - we have an answer
            if self._verbose:
                self._logger.info(
                    "Final answer reached",
                    response=reasoning_step.response if hasattr(reasoning_step, "response") else llm_output,
                )

            return AgentOutput(
                response=ChatMessage(role="assistant", content=llm_output or ""),
                current_agent_name=self.name,
                raw=response.raw if hasattr(response, "raw") else None,
            )

        # Action step - need to execute tool
        if isinstance(reasoning_step, ActionReasoningStep):
            if self._verbose:
                self._logger.debug(
                    "Executing tool action", tool_name=reasoning_step.action, tool_input=reasoning_step.action_input
                )

            # Create tool selection
            tool_selection = ToolSelection(
                tool_id=str(uuid.uuid4()), tool_name=reasoning_step.action, tool_kwargs=reasoning_step.action_input
            )

            return AgentOutput(
                response=response.message,
                tool_calls=[tool_selection],
                current_agent_name=self.name,
                raw=response.raw if hasattr(response, "raw") else None,
            )

        # Shouldn't reach here, but continue if we do
        return AgentOutput(
            response=response.message,
            current_agent_name=self.name,
            raw=response.raw if hasattr(response, "raw") else None,
        )

    async def handle_tool_call_results(self, ctx: Context, results: List[ToolCallResult], memory: BaseMemory) -> None:
        """Handle tool call results by adding observations to reasoning."""
        # Get current reasoning from context
        current_reasoning: List[BaseReasoningStep] = await ctx.store.get(CTX_CURRENT_REASONING, default=[])

        # Get sources from context
        sources: List[Any] = await ctx.store.get(CTX_SOURCES, default=[])

        # Process each tool result
        for result in results:
            observation = str(result.tool_output.content)

            if self._verbose:
                self._logger.debug(
                    "Tool execution completed",
                    tool_name=result.tool_name,
                    output_preview=observation[:100] if len(observation) > 100 else observation,
                    is_error=result.tool_output.is_error,
                )

            # Add observation to reasoning
            obs_step = ObservationReasoningStep(observation=observation, return_direct=result.return_direct)
            current_reasoning.append(obs_step)

            # Track sources if not an error
            if not result.tool_output.is_error:
                # Try to preserve the original type if possible
                try:
                    # If the content looks like a number, convert it
                    if isinstance(result.tool_output.content, str) and result.tool_output.content.isdigit():
                        sources.append(int(result.tool_output.content))
                    else:
                        sources.append(result.tool_output.content)
                except Exception:
                    sources.append(result.tool_output.content)

            # If return_direct, add response step
            if result.return_direct and not result.tool_output.is_error:
                current_reasoning.append(
                    ResponseReasoningStep(thought=observation, response=observation, is_streaming=False)
                )
                break

        # Update context
        await ctx.store.set(CTX_CURRENT_REASONING, current_reasoning)
        await ctx.store.set(CTX_SOURCES, sources)

    async def finalize(self, ctx: Context, output: AgentOutput, memory: BaseMemory) -> AgentOutput:
        """Store reasoning chain in memory and clean up response format."""
        # Get current reasoning from context
        current_reasoning: List[BaseReasoningStep] = await ctx.store.get(CTX_CURRENT_REASONING, default=[])

        # If we have reasoning steps and the last one is a response
        if current_reasoning and isinstance(current_reasoning[-1], ResponseReasoningStep):
            # Create reasoning string for memory
            reasoning_str = "\\n".join([step.get_content() for step in current_reasoning])

            if reasoning_str:
                # Store in memory
                reasoning_msg = ChatMessage(role="assistant", content=reasoning_str)
                await memory.aput(reasoning_msg)

            # Clean up response - remove "Answer:" prefix if present
            if output.response.content and "Answer:" in output.response.content:
                start_idx = output.response.content.find("Answer:")
                if start_idx != -1:
                    output.response.content = output.response.content[start_idx + len("Answer:") :].strip()

        # Store reasoning and sources for later retrieval in run()
        await ctx.store.set("final_reasoning", current_reasoning)
        await ctx.store.set("final_sources", await ctx.store.get(CTX_SOURCES, default=[]))

        # Clear reasoning context for next interaction
        await ctx.store.set(CTX_CURRENT_REASONING, [])
        await ctx.store.set(CTX_SOURCES, [])

        return output

    def _get_context(self, handler: Any) -> Optional[Context]:
        """Get context from handler using multiple fallback strategies."""
        # Try direct context attribute (most common)
        if hasattr(handler, "ctx"):
            return cast(Optional[Context], handler.ctx)

        # Try private context attribute
        if hasattr(handler, "_context"):
            return cast(Optional[Context], handler._context)

        # Try context via workflow reference
        if hasattr(handler, "workflow") and hasattr(handler.workflow, "_context"):
            return cast(Optional[Context], handler.workflow._context)

        # Try context from self (workflow instance)
        if hasattr(self, "_context"):
            return cast(Optional[Context], getattr(self, "_context", None))

        return None

    async def get_results_from_handler(self, handler: Any) -> dict[str, Any]:
        """Extract results from a WorkflowHandler for testing and production use."""

        # Process all events
        async for _ in handler.stream_events():
            pass

        # Get the final result
        result = await handler

        # Extract the response from AgentOutput
        if hasattr(result, "response"):
            response_content = result.response.content if hasattr(result.response, "content") else str(result.response)
        else:
            response_content = str(result)

        # Get the workflow context to retrieve reasoning and sources
        reasoning = []
        sources = []

        # Try to access the handler's context using helper method
        try:
            ctx = self._get_context(handler)

            if ctx:
                # Try to get final values stored in finalize
                reasoning = await ctx.store.get("final_reasoning", default=[])
                sources = await ctx.store.get("final_sources", default=[])

                # If not found, try the regular keys
                if not reasoning:
                    reasoning = await ctx.store.get(CTX_CURRENT_REASONING, default=[])
                if not sources:
                    sources = await ctx.store.get(CTX_SOURCES, default=[])
        except Exception as e:
            if self._verbose:
                logger.warning("Could not retrieve context data", error=str(e))

        # Get memory for chat history - handler might have memory in its context
        chat_history: List[ChatMessage] = []
        try:
            if hasattr(handler, "workflow") and hasattr(handler.workflow, "_memory"):
                memory = handler.workflow._memory
                if memory:
                    chat_history = await memory.aget()
        except Exception:
            pass

        return {"response": response_content, "sources": sources, "reasoning": reasoning, "chat_history": chat_history}


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
