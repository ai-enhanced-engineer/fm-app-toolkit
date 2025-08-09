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
        llm: LLM,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 5,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialize the ReAct agent workflow.
        
        Args:
            llm: The language model to use for reasoning
            tools: List of available tools
            max_iterations: Maximum reasoning iterations before stopping
            verbose: Whether to print reasoning steps
            **kwargs: Additional workflow arguments
        """
        super().__init__(**kwargs)
        
        self.llm = llm
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Create tool registry
        self.tool_registry = {tool.name: tool for tool in self.tools}
        
        # Initialize memory buffer
        self.memory_buffer = ChatMemoryBuffer.from_defaults(llm=llm)
        
        # Create a simpler system prompt without tool placeholders
        # The formatter will add tool information automatically
        system_prompt = """You are a helpful assistant that provides clear and concise answers. You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses."""
        
        self.formatter = ReActChatFormatter(
            system_header=system_prompt,
            context=""  # No additional context for simplicity
        )
        
        # Initialize parser - using LlamaIndex's built-in ReActOutputParser
        self.parser = ReActOutputParser()
        
        # Track sources and reasoning
        self.sources: List[Any] = []
        self.current_reasoning: List[BaseReasoningStep] = []
        
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
                chat_history=self.memory_buffer.get()
            )
            
        # Add user message to memory
        self.memory_buffer.put(ChatMessage(role="user", content=user_query))
        
        # Reset for new query
        self.sources = []
        self.current_reasoning = []
        
        # Store query in context for later use
        await ctx.set("user_query", user_query)
        await ctx.set("iteration_count", 0)
        
        if self.verbose:
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
        # Check iteration count
        iteration_count = await ctx.get("iteration_count", default=0)
        
        if iteration_count >= self.max_iterations:
            return stop_workflow(
                response="I couldn't complete the reasoning in the allowed iterations.",
                sources=self.sources,
                reasoning=self.current_reasoning,
                chat_history=self.memory_buffer.get()
            )
            
        # Get chat history
        chat_history = self.memory_buffer.get()
        
        # Create simple tool metadata for the formatter
        # Using a simple dict format that the formatter can handle
        from llama_index.core.tools import FunctionTool
        
        function_tools = []
        for tool in self.tools:
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
        formatted_messages = self.formatter.format(
            function_tools,
            chat_history,
            current_reasoning=self.current_reasoning if self.current_reasoning else None
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
        
        # Update iteration count
        iteration_count = await ctx.get("iteration_count", default=0)
        await ctx.set("iteration_count", iteration_count + 1)
        
        if self.verbose:
            print(f"\n--- Iteration {iteration_count + 1} ---")
            
        # Get LLM response
        response = await self.llm.achat(messages)
        llm_output = response.message.content
        
        if self.verbose:
            print(f"LLM Output:\n{llm_output}")
            
        # Parse the output using the built-in ReActOutputParser
        try:
            reasoning_step = self.parser.parse(llm_output or "")
            
            if isinstance(reasoning_step, ActionReasoningStep):
                # Action step - execute tool
                self.current_reasoning.append(reasoning_step)
                
                # Create tool selection
                tool_selection = ToolSelection(
                    tool_id=f"tool_{iteration_count}",
                    tool_name=reasoning_step.action,
                    tool_kwargs=reasoning_step.action_input
                )
                
                return ToolCallEvent(tool_calls=[tool_selection])
                
            elif isinstance(reasoning_step, ResponseReasoningStep):
                # Response step - we have an answer
                self.current_reasoning.append(reasoning_step)
                
                # Store assistant response in memory
                self.memory_buffer.put(ChatMessage(role="assistant", content=llm_output))
                
                return stop_workflow(
                    response=reasoning_step.response,
                    sources=self.sources,
                    reasoning=self.current_reasoning,
                    chat_history=self.memory_buffer.get()
                )
                
            else:
                # Other reasoning step (e.g., ObservationReasoningStep) - continue
                if self.verbose:
                    print(f"Continuing reasoning with step type: {type(reasoning_step).__name__}")
                self.current_reasoning.append(reasoning_step)
                return PrepEvent()
                
        except ValueError as e:
            # Parser couldn't parse the output - this might be a thought-only response
            if self.verbose:
                print(f"Parser error (continuing): {e}")
            # Continue with next iteration
            return PrepEvent()
        except Exception as e:
            if self.verbose:
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
        for tool_call in ev.tool_calls:
            tool_name = tool_call.tool_name
            tool_kwargs = tool_call.tool_kwargs
            
            if tool_name not in self.tool_registry:
                observation = f"Error: Tool '{tool_name}' not found"
            else:
                try:
                    tool = self.tool_registry[tool_name]
                    result = tool.function(**tool_kwargs)
                    observation = str(result)
                    self.sources.append(result)
                except Exception as e:
                    observation = f"Error executing tool: {str(e)}"
                    
            if self.verbose:
                print(f"Tool Output: {observation}")
                
            # Add observation to reasoning
            observation_step = ObservationReasoningStep(observation=observation)
            self.current_reasoning.append(observation_step)
            
        return PrepEvent()
        
    async def run(self, user_msg: str) -> str:  # type: ignore[override]
        """Run the workflow with a user message.
        
        This is a convenience method that starts the workflow and returns the result.
        
        Args:
            user_msg: The user's query
            
        Returns:
            The agent's response string
        """
        result = await super().run(user_msg=user_msg)
        
        # Extract just the response for backward compatibility
        if isinstance(result, dict) and "response" in result:
            # For simple use, return just the response text
            # But the full result is available if needed
            return str(result.get("response", ""))
        return str(result)