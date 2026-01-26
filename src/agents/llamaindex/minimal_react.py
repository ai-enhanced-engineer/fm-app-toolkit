"""Minimal educational ReAct agent implementation for learning purposes.

This module demonstrates the core ReAct (Reasoning + Acting) pattern in a concise,
understandable implementation. It's designed for pedagogy, not production use.

ReAct Pattern (Yao et al., 2022): https://arxiv.org/abs/2210.03629

Key differences from simple_react.py:
- No hidden base class logic - the loop is explicit and visible
- Inline prompt template - no abstraction layers
- Simple regex parsing - no complex parser framework
- Heavily commented for learning

For production use, see simple_react.py which uses LlamaIndex's BaseWorkflowAgent
and provides better error handling, memory management, and observability.
"""

import json
import re
from typing import Any

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import LLM

from src.agents.common import Tool


class ToolExecutionError(Exception):
    """Raised when a tool execution fails.

    Provides structured error information including the tool name and original error.
    """

    def __init__(self, tool_name: str, original_error: Exception) -> None:
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Error executing {tool_name}: {original_error}")


# The ReAct prompt template - visible and inline for learning
REACT_PROMPT = """You are an AI assistant that uses tools to answer questions.

Available tools:
{tool_descriptions}

Use this EXACT format for your responses:

Thought: [Your reasoning about what to do]
Action: [Tool name to use]
Action Input: {{"arg1": "value1", "arg2": "value2"}}

After observing tool output:
Thought: [Your reasoning about the result]
Action: [Another tool] OR Answer: [Final answer to user]

When you have enough information:
Thought: [Final reasoning]
Answer: [Complete answer to user's question]

IMPORTANT:
- Use "Action Input" with JSON format for tool arguments
- Only use tools that are listed above
- Provide your final answer after "Answer:"

Previous conversation:
{chat_history}

Current question: {query}
{observations}"""


class MinimalReActAgent:
    """A minimal ReAct agent with explicit control flow for learning.

    This implementation makes the ReAct loop visible:
    1. Format prompt with tools and query
    2. Get LLM response
    3. Parse for Thought/Action/Answer
    4. Execute tool if Action present
    5. Add observation and loop
    6. Return when Answer present or max steps reached

    Example:
        >>> from src.agents.llamaindex.sample_tools import calculate, get_weather
        >>> tools = [
        ...     Tool("calculate", "Do math", calculate),
        ...     Tool("get_weather", "Get weather for a city", get_weather)
        ... ]
        >>> agent = MinimalReActAgent(llm, tools, max_steps=5)
        >>> result = await agent.run("What's the weather in Tokyo?")
    """

    def __init__(self, llm: LLM, tools: list[Tool], max_steps: int = 10, verbose: bool = False) -> None:
        """Initialize the agent with an LLM and tools.

        Args:
            llm: Language model to use for reasoning
            tools: List of tools the agent can use
            max_steps: Maximum reasoning iterations before giving up
            verbose: If True, print step-by-step execution for transparency
        """
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}  # Index by name for lookup
        self.max_steps = max_steps
        self.verbose = verbose

    def _format_tools(self) -> str:
        """Format tool descriptions for the prompt."""
        return "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])

    def _parse_response(self, text: str) -> tuple[str | None, str | None, dict[str, Any] | None, str | None]:
        """Parse LLM response for Thought/Action/Action Input/Answer.

        Returns:
            (thought, action, action_input, answer) tuple where each can be None
        """
        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Answer):|$)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None

        # Extract action
        action_match = re.search(r"Action:\s*(\w+)", text)
        action = action_match.group(1).strip() if action_match else None

        # Extract action input (JSON format)
        action_input = None
        if action:
            # Match {} for empty JSON or {...} for non-empty
            input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
            if input_match:
                try:
                    action_input = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    action_input = {}
            else:
                # If no Action Input found, default to empty dict
                action_input = {}

        # Extract answer
        answer_match = re.search(r"Answer:\s*(.+?)$", text, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else None

        return thought, action, action_input, answer

    def _execute_tool(self, name: str, input_args: dict[str, Any]) -> str:
        """Execute a tool and return its output as a string.

        Raises:
            ToolExecutionError: When tool execution fails (caught internally for error message).
        """
        if name not in self.tools:
            return f"Error: Tool '{name}' not found. Available tools: {list(self.tools.keys())}"

        try:
            tool = self.tools[name]
            result = tool.function(**input_args)
            return str(result)
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            raise ToolExecutionError(name, e) from e
        except Exception as e:
            # Catch-all for unexpected errors, but still wrap in structured error
            raise ToolExecutionError(name, e) from e

    async def run(self, query: str) -> dict[str, Any]:
        """Run the agent on a query using the explicit ReAct loop.

        This is THE CORE LOOP - read this carefully to understand ReAct:

        Args:
            query: User's question or request

        Returns:
            Dictionary with 'response', 'reasoning' (list of steps), 'sources' (tool outputs)
        """
        # Track the conversation for context
        chat_history: list[ChatMessage] = []
        observations: list[str] = []  # Use list for efficient string building
        reasoning_steps: list[str] = []
        sources: list[str] = []

        # THE EXPLICIT REACT LOOP
        for step in range(self.max_steps):
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Step {step + 1}/{self.max_steps}")
                print(f"{'=' * 60}")

            # 1. Format the prompt with current context
            prompt_text = REACT_PROMPT.format(
                tool_descriptions=self._format_tools(),
                chat_history="\n".join([f"{msg.role}: {msg.content}" for msg in chat_history]),
                query=query,
                observations="\n".join(observations),
            )

            # 2. Get LLM response
            messages = [ChatMessage(role="user", content=prompt_text)]
            response = await self.llm.achat(messages)
            llm_output = response.message.content or ""

            # Store for reasoning trace
            reasoning_steps.append(llm_output)

            # 3. Parse the response
            thought, action, action_input, answer = self._parse_response(llm_output)

            if self.verbose and thought:
                print(f"💭 Thought: {thought}")

            # 4. If we have an answer, we're done!
            if answer:
                if self.verbose:
                    print(f"✅ Answer: {answer}\n")
                return {"response": answer, "reasoning": reasoning_steps, "sources": sources}

            # 5. If we have an action, execute the tool
            if action and action_input is not None:
                if self.verbose:
                    print(f"🔧 Action: {action}")
                    print(f"📝 Input: {action_input}")
                try:
                    observation = self._execute_tool(action, action_input)
                except ToolExecutionError as e:
                    observation = str(e)
                if self.verbose:
                    print(f"👀 Observation: {observation}")
                observations.append(f"Observation: {observation}")
                sources.append(observation)
            else:
                # No valid action or answer - ask LLM to continue
                if self.verbose:
                    print("⚠️  No valid action or answer detected")
                observations.append("Observation: Please provide either an Action or Answer.")

        # Max steps reached without answer
        return {
            "response": "I couldn't complete the task within the allowed steps.",
            "reasoning": reasoning_steps,
            "sources": sources,
        }
