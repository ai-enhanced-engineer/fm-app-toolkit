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

import logging
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
from llama_index.core.workflow.handler import WorkflowHandler

logger = logging.getLogger(__name__)

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
                    function_tools.append(
                        FunctionTool.from_defaults(fn=tool.function, name=tool.name, description=tool.description)
                    )
                except Exception as e:
                    logger.warning("Failed to register tool '%s': %s", tool.name, e)
                    if verbose:
                        print(f"Warning: Failed to register tool '{tool.name}': {e}")

        # Combine system header and extra context for system prompt
        system_prompt = f"{system_header}\n\n{extra_context}" if extra_context else system_header

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
        self._extra_context = extra_context or ""
        self._formatter = ReActChatFormatter(system_header=self._system_header, context=self._extra_context)

        # Initialize parser - using LlamaIndex's built-in ReActOutputParser
        self._output_parser = ReActOutputParser()

    async def take_step(
        self,
        ctx: Context,
        llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
        _memory: BaseMemory,
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

        # Check if we've exceeded max reasoning steps
        if len(current_reasoning) >= self._max_reasoning:
            if self._verbose:
                print(f"⚠️ Exceeded max reasoning steps: {len(current_reasoning)}/{self._max_reasoning}")
            error_msg = "I couldn't complete the reasoning in the allowed iterations."
            current_reasoning.append(
                ResponseReasoningStep(thought="Exceeded max reasoning steps", response=error_msg, is_streaming=False)
            )
            await ctx.store.set(CTX_CURRENT_REASONING, current_reasoning)
            return AgentOutput(
                response=ChatMessage(role="assistant", content=error_msg), current_agent_name=self.name, raw=None
            )

        # Format input using ReActChatFormatter
        formatted_messages = self._formatter.format(tools, llm_input, current_reasoning=current_reasoning or None)

        if self._verbose:
            print(f"💭 Step {len(current_reasoning) + 1}: Starting reasoning...")

        # Get LLM response
        response = await self.llm.achat(formatted_messages)
        llm_output = response.message.content

        if self._verbose:
            preview = llm_output[:100] if llm_output else ""
            print(f"📝 LLM output: {preview}{'...' if llm_output and len(llm_output) > 100 else ''}")

        # Parse the output
        try:
            reasoning_step = self._output_parser.parse(llm_output or "", is_streaming=False)
        except ValueError as e:
            if self._verbose:
                print(f"⚠️ Parse error: {e}")

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
            if self._verbose:
                print("✅ Final answer reached")
            return AgentOutput(
                response=ChatMessage(role="assistant", content=llm_output or ""),
                current_agent_name=self.name,
                raw=response.raw if hasattr(response, "raw") else None,
            )

        # Action step - need to execute tool
        if isinstance(reasoning_step, ActionReasoningStep):
            if self._verbose:
                print(f"🔧 Tool call: {reasoning_step.action}")

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

    async def handle_tool_call_results(self, ctx: Context, results: List[ToolCallResult], _memory: BaseMemory) -> None:
        """Handle tool call results by adding observations to reasoning."""
        # Get current reasoning from context
        current_reasoning: List[BaseReasoningStep] = await ctx.store.get(CTX_CURRENT_REASONING, default=[])

        # Get sources from context
        sources: List[Any] = await ctx.store.get(CTX_SOURCES, default=[])

        # Process each tool result
        for result in results:
            observation = str(result.tool_output.content)

            if self._verbose:
                preview = observation[:100] if len(observation) > 100 else observation
                status = "❌ Error" if result.tool_output.is_error else "✅ Success"
                print(f"{status}: {result.tool_name} → {preview}")

            # Add observation to reasoning
            obs_step = ObservationReasoningStep(observation=observation, return_direct=result.return_direct)
            current_reasoning.append(obs_step)

            # Track sources if not an error
            if not result.tool_output.is_error:
                # Convert string numbers to int for type preservation
                content = result.tool_output.content
                if isinstance(content, str) and content.isdigit():
                    sources.append(int(content))
                else:
                    sources.append(content)

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
                output.response.content = output.response.content.split("Answer:", 1)[-1].strip()

        # Store reasoning and sources for later retrieval in run()
        await ctx.store.set("final_reasoning", current_reasoning)
        await ctx.store.set("final_sources", await ctx.store.get(CTX_SOURCES, default=[]))

        # Clear reasoning context for next interaction
        await ctx.store.set(CTX_CURRENT_REASONING, [])
        await ctx.store.set(CTX_SOURCES, [])

        return output

    async def get_results_from_handler(self, handler: WorkflowHandler) -> dict[str, Any]:
        """Extract results from a WorkflowHandler for testing and production use."""
        # Process all events
        async for _ in handler.stream_events():
            pass

        # Get the final result
        result = await handler

        # Extract the response
        if hasattr(result, "response"):
            response_content = result.response.content if hasattr(result.response, "content") else str(result.response)
        else:
            response_content = str(result)

        # Get reasoning and sources from context
        reasoning = []
        sources = []
        if hasattr(handler, "ctx") and handler.ctx:
            ctx = cast(Optional[Context], handler.ctx)
            reasoning = await ctx.store.get("final_reasoning", default=[])
            sources = await ctx.store.get("final_sources", default=[])

        # Get chat history from memory
        chat_history: List[ChatMessage] = []
        if hasattr(handler, "workflow") and hasattr(handler.workflow, "_memory"):
            memory = handler.workflow._memory
            if memory:
                chat_history = await memory.aget()

        return {"response": response_content, "sources": sources, "reasoning": reasoning, "chat_history": chat_history}
