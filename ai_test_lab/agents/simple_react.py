"""A minimalistic ReAct agent implementation using Workflow pattern.

This module demonstrates the core ReAct (Reasoning + Acting) pattern using
llama_index's Workflow architecture:
1. Thought: The agent reasons about what to do
2. Action: The agent decides to use a tool (optional)
3. Observation: The agent observes the tool's output
4. Answer: The agent provides a final response

This is a simplified version focusing on the reasoning loop and action calling,
designed for educational purposes.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from llama_index.core.agent.react import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import ToolSelection
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from .events import InputEvent, PrepEvent, ToolCallEvent, stop_workflow

# Context key for storing current reasoning steps
CTX_CURRENT_REASONING = "current_reasoning"


@dataclass
class Tool:
    """Simple tool representation."""
    name: str
    function: Callable[..., Any]
    description: str


class SimpleReActAgent(Workflow):
    """A minimalistic ReAct agent using the Workflow pattern.
    
    This agent follows the ReAct pattern using workflow steps:
    - Initialize context and format input
    - Get LLM response and parse it
    - Execute tools if needed
    - Return final answer
    
    This is designed for educational purposes to demonstrate the core
    concepts without production complexity.
    """
    
    def __init__(
        self,
        *args: Any,
        llm: LLM,
        system_header: str,
        extra_context: Optional[str] = None,
        max_reasoning: Optional[int] = None,
        tools: Optional[List[Tool]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialize the ReAct agent workflow.
        
        Args:
            llm: The language model to use for reasoning
            system_header: The system prompt/header for the agent
            extra_context: Additional context to include in prompts
            max_reasoning: Maximum reasoning iterations before stopping (default: 15)
            tools: List of available tools
            chat_history: Optional initial chat history
            verbose: Whether to print reasoning steps
            **kwargs: Additional workflow arguments
        """
        super().__init__(*args, **kwargs)
        
        # Core components
        self._llm = llm
        self._system_header = system_header
        self._extra_context = extra_context if extra_context is not None else ""
        self._max_reasoning = max_reasoning if max_reasoning is not None else 15
        self._tools = tools if tools is not None else []
        self._verbose = verbose
        
        # Create tool registry
        self._tool_registry = {tool.name: tool for tool in self._tools}
        
        # Initialize memory buffer with optional chat history
        self._memory_buffer = ChatMemoryBuffer.from_defaults(
            llm=llm, 
            chat_history=chat_history
        )
        
        # Initialize formatter with system header and extra context
        self._formatter = ReActChatFormatter(
            system_header=self._system_header,
            context=self._extra_context
        )
        
        # Initialize parser - using LlamaIndex's built-in ReActOutputParser
        self._output_parser = ReActOutputParser()
        
        # Track sources for the current query
        self._sources: List[Any] = []
        
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> PrepEvent | StopEvent:
        """Initialize the workflow and process the user input.
        
        Args:
            ctx: The workflow context
            ev: The start event containing the user query
            
        Returns:
            PrepEvent to continue or StopEvent if no input
        """
        # Get user query from the event
        user_query = ev.get("user_msg", "")
        
        if not user_query:
            return stop_workflow(
                response="",
                sources=[],
                reasoning=[],
                chat_history=self._memory_buffer.get()
            )
            
        # Add user message to memory
        self._memory_buffer.put(ChatMessage(role="user", content=user_query))
        
        # Reset sources for new query
        self._sources = []
        
        # Initialize current reasoning in context
        await ctx.set(CTX_CURRENT_REASONING, [])
        
        # Store query in context for later use
        await ctx.set("user_query", user_query)
        await ctx.set("iteration_count", 0)
        
        if self._verbose:
            print(f"Processing query: {user_query}")
            
        return PrepEvent()
        
    @step
    async def format_llm_input(self, ctx: Context, ev: PrepEvent) -> InputEvent | StopEvent:
        """Format the input for the LLM using ReActChatFormatter.
        
        Args:
            ctx: The workflow context
            ev: The prep event
            
        Returns:
            InputEvent with formatted messages or StopEvent if max iterations reached
        """
        # Get current reasoning from context
        current_reasoning: List[BaseReasoningStep] = await ctx.get(CTX_CURRENT_REASONING, default=[])
        
        # Check if we've exceeded max reasoning steps
        if len(current_reasoning) >= self._max_reasoning:
            current_reasoning.append(
                ResponseReasoningStep(
                    thought="Exceeded max reasoning steps",
                    response="I couldn't complete the reasoning in the allowed iterations."
                )
            )
            return stop_workflow(
                response="I couldn't complete the reasoning in the allowed iterations.",
                sources=self._sources,
                reasoning=current_reasoning,
                chat_history=self._memory_buffer.get()
            )
            
        # Get chat history
        chat_history = self._memory_buffer.get()
        
        # Create simple tool metadata for the formatter
        from llama_index.core.tools import FunctionTool
        
        function_tools = []
        for tool in self._tools:
            # Create FunctionTool for proper formatting
            try:
                func_tool = FunctionTool.from_defaults(
                    fn=tool.function,
                    name=tool.name,
                    description=tool.description
                )
                function_tools.append(func_tool)
            except Exception:
                # If FunctionTool creation fails, skip this tool
                pass
        
        # Format the messages
        formatted_messages = self._formatter.format(
            function_tools,
            chat_history,
            current_reasoning=current_reasoning if current_reasoning else None
        )
        
        return InputEvent(input=formatted_messages)
        
    @step
    async def get_llm_response(self, ctx: Context, ev: InputEvent) -> ToolCallEvent | PrepEvent | StopEvent:
        """Get response from the LLM and parse it.
        
        Args:
            ctx: The workflow context
            ev: The input event with formatted messages
            
        Returns:
            ToolCallEvent if action needed, PrepEvent to continue, or StopEvent with answer
        """
        messages = ev.input
        
        # Get current reasoning from context
        current_reasoning: List[BaseReasoningStep] = await ctx.get(CTX_CURRENT_REASONING, default=[])
        
        # Update iteration count
        iteration_count = await ctx.get("iteration_count", default=0)
        await ctx.set("iteration_count", iteration_count + 1)
        
        if self._verbose:
            print(f"\n--- Iteration {iteration_count + 1} ---")
            
        # Get LLM response
        response = await self._llm.achat(messages)
        llm_output = response.message.content
        
        if self._verbose:
            print(f"LLM Output:\n{llm_output}")
            
        # Parse the output using the built-in ReActOutputParser
        try:
            reasoning_step = self._output_parser.parse(llm_output or "")
            
            if isinstance(reasoning_step, ActionReasoningStep):
                # Action step - execute tool
                current_reasoning.append(reasoning_step)
                await ctx.set(CTX_CURRENT_REASONING, current_reasoning)
                
                # Create tool selection
                tool_selection = ToolSelection(
                    tool_id=f"tool_{iteration_count}",
                    tool_name=reasoning_step.action,
                    tool_kwargs=reasoning_step.action_input
                )
                
                return ToolCallEvent(tool_calls=[tool_selection])
                
            elif isinstance(reasoning_step, ResponseReasoningStep):
                # Response step - we have an answer
                current_reasoning.append(reasoning_step)
                await ctx.set(CTX_CURRENT_REASONING, current_reasoning)
                
                # Store assistant response in memory
                self._memory_buffer.put(ChatMessage(role="assistant", content=llm_output))
                
                return stop_workflow(
                    response=reasoning_step.response,
                    sources=self._sources,
                    reasoning=current_reasoning,
                    chat_history=self._memory_buffer.get()
                )
                
            else:
                # Other reasoning step (e.g., ObservationReasoningStep) - continue
                if self._verbose:
                    print(f"Continuing reasoning with step type: {type(reasoning_step).__name__}")
                current_reasoning.append(reasoning_step)
                await ctx.set(CTX_CURRENT_REASONING, current_reasoning)
                return PrepEvent()
                
        except ValueError as e:
            # Parser couldn't parse the output - this might be a thought-only response
            if self._verbose:
                print(f"Parser error (continuing): {e}")
            # Continue with next iteration
            return PrepEvent()
        except Exception as e:
            if self._verbose:
                print(f"Error parsing LLM output: {e}")
            # Continue with next iteration
            return PrepEvent()
            
    @step
    async def execute_tool(self, ctx: Context, ev: ToolCallEvent) -> PrepEvent:
        """Execute the requested tools.
        
        Args:
            ctx: The workflow context
            ev: The tool call event
            
        Returns:
            PrepEvent to continue reasoning
        """
        # Get current reasoning from context
        current_reasoning: List[BaseReasoningStep] = await ctx.get(CTX_CURRENT_REASONING, default=[])
        
        for tool_call in ev.tool_calls:
            tool_name = tool_call.tool_name
            tool_kwargs = tool_call.tool_kwargs
            
            if tool_name not in self._tool_registry:
                observation = f"Error: Tool '{tool_name}' not found"
            else:
                try:
                    tool = self._tool_registry[tool_name]
                    result = tool.function(**tool_kwargs)
                    observation = str(result)
                    self._sources.append(result)
                except Exception as e:
                    observation = f"Error executing tool: {str(e)}"
                    
            if self._verbose:
                print(f"Tool Output: {observation}")
                
            # Add observation to reasoning
            observation_step = ObservationReasoningStep(observation=observation)
            current_reasoning.append(observation_step)
            
        # Update reasoning in context
        await ctx.set(CTX_CURRENT_REASONING, current_reasoning)
            
        return PrepEvent()
        
    async def run(self, user_msg: str, **kwargs: Any) -> str:  # type: ignore[override]
        """Run the workflow with a user message.
        
        This is a convenience method that starts the workflow and returns the result.
        
        Args:
            user_msg: The user's query
            **kwargs: Additional keyword arguments
            
        Returns:
            The agent's response string
        """
        result = await super().run(user_msg=user_msg, **kwargs)
        
        # Extract just the response for backward compatibility
        if isinstance(result, dict) and "response" in result:
            # For simple use, return just the response text
            # But the full result is available if needed
            return str(result.get("response", ""))
        return str(result)