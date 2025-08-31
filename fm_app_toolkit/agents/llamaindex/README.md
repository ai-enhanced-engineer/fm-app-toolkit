# Agents Module - ReAct Pattern Implementation

## What Are Agents, Really?

Everyone talks about agents, but what does that actually look like in code? Where do they live in your application architecture? How do they integrate with your business logic?

An agent is fundamentally an orchestration layer that:
1. **Receives** a user request
2. **Reasons** about what steps to take
3. **Acts** by calling tools or functions
4. **Observes** the results
5. **Iterates** until it can provide a complete answer

This module provides a concrete, production-ready implementation of the ReAct (Reasoning + Acting) pattern that shows exactly how agents work in practice.

## The ReAct Pattern

The ReAct pattern, introduced in the paper ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629), provides a structured approach to agent reasoning:

```
User Query → Thought → Action → Observation → Thought → ... → Answer
```

Each step serves a purpose:
- **Thought**: The agent reasons about what to do next
- **Action**: The agent decides to use a tool (optional)
- **Observation**: The agent sees the tool's output
- **Answer**: The agent provides the final response

This pattern is powerful because it:
- Makes reasoning transparent and debuggable
- Allows for complex multi-step problem solving
- Provides clear integration points for tools
- Maintains a clear audit trail of decisions

## Module Organization

```
agents/
├── simple_react.py           # Main ReAct agent implementation
├── events.py                # Event helpers for workflow coordination
├── sample_tools.py          # Example tools for testing
└── react_prompt_template.md # ReAct prompt template
```

### SimpleReActAgent

Our implementation extends `BaseWorkflowAgent` from LlamaIndex, providing:
- Clear, pedagogical code that's easy to understand
- Production-ready patterns (WorkflowHandler)
- Full compatibility with the LlamaIndex ecosystem
- Deterministic testing with mock LLMs

## How It Works

### 1. Agent Initialization

```python
from fm_app_toolkit.agents.llamaindex import SimpleReActAgent
from fm_app_toolkit.tools import create_calculator_tool

agent = SimpleReActAgent(
    llm=llm,                    # Any LlamaIndex LLM (real or mock)
    tools=[calculator_tool],    # List of available tools
    max_iterations=10,          # Maximum reasoning steps
    verbose=True               # Show reasoning process
)
```

### 2. The Reasoning Loop

When you run the agent, it enters a reasoning loop:

```python
# Start the agent
handler = agent.run(user_msg="What is 15 * 7 + 23?")

# The agent internally:
# 1. Formats the prompt with ReAct instructions
# 2. Sends to LLM: "You need to calculate 15 * 7 + 23..."
# 3. Receives: "Thought: I need to multiply first.\nAction: multiply\nAction Input: {'a': 15, 'b': 7}"
# 4. Executes the multiply tool
# 5. Observes: "105"
# 6. Continues: "Thought: Now add 23.\nAction: add\nAction Input: {'a': 105, 'b': 23}"
# 7. Observes: "128"
# 8. Concludes: "Answer: 15 * 7 + 23 = 128"
```

### 3. Result Extraction

The agent returns a WorkflowHandler, following LlamaIndex production patterns:

```python
# Extract structured results
result = await agent.get_results_from_handler(handler)

print(result["response"])    # "15 * 7 + 23 = 128"
print(result["sources"])     # [105, 128] - tool outputs
print(result["reasoning"])   # List of reasoning steps
print(result["chat_history"]) # Conversation history
```

## Complete Usage Examples

### Basic Math Agent

```python
from fm_app_toolkit.agents.llamaindex import SimpleReActAgent
from fm_app_toolkit.tools import create_multiply_tool, create_add_tool

# Create tools
multiply = create_multiply_tool()
add = create_add_tool()

# Create agent with real LLM
from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-4")

agent = SimpleReActAgent(
    llm=llm,
    tools=[multiply, add],
    verbose=True
)

# Run the agent
handler = agent.run(user_msg="Calculate (12 * 5) + 30")
result = await agent.get_results_from_handler(handler)

print(f"Answer: {result['response']}")
# Answer: (12 * 5) + 30 = 90
```

### Testing with Mock LLMs

```python
from fm_app_toolkit.testing import MockLLMWithChain
from fm_app_toolkit.agents.llamaindex import SimpleReActAgent

# Define deterministic agent behavior
mock_llm = MockLLMWithChain(chain=[
    "Thought: First multiply 12 by 5.\nAction: multiply\nAction Input: {'a': 12, 'b': 5}",
    "Thought: Got 60. Now add 30.\nAction: add\nAction Input: {'a': 60, 'b': 30}",
    "Thought: The result is 90.\nAnswer: (12 * 5) + 30 = 90"
])

# Create agent with mock
agent = SimpleReActAgent(
    llm=mock_llm,
    tools=[multiply, add]
)

# Test the agent
handler = agent.run(user_msg="Calculate (12 * 5) + 30")
result = await agent.get_results_from_handler(handler)

assert result["response"] == "(12 * 5) + 30 = 90"
assert result["sources"] == [60, 90]  # Tool outputs
```

### Information Retrieval Agent

```python
from fm_app_toolkit.agents.llamaindex import SimpleReActAgent
from llama_index.core.tools import FunctionTool

def search_database(query: str) -> str:
    """Search the company database."""
    # Simulate database search
    return f"Found 3 results for '{query}'"

def get_details(id: str) -> str:
    """Get details for a specific item."""
    return f"Details for item {id}: Product specification..."

# Create tools
search_tool = FunctionTool.from_defaults(fn=search_database)
details_tool = FunctionTool.from_defaults(fn=get_details)

# Create agent
agent = SimpleReActAgent(
    llm=llm,
    tools=[search_tool, details_tool]
)

# Run complex query
handler = agent.run(user_msg="Find products related to 'wireless' and get details for the first one")
result = await agent.get_results_from_handler(handler)
```

## Testing Strategies

### 1. Deterministic Multi-Step Testing

```python
import pytest
from fm_app_toolkit.agents.llamaindex import SimpleReActAgent
from fm_app_toolkit.testing import MockLLMWithChain

@pytest.mark.asyncio
async def test_multi_step_calculation():
    """Test agent performs calculations in correct order."""
    
    chain = [
        "Thought: Calculate 10 * 3.\nAction: multiply\nAction Input: {'a': 10, 'b': 3}",
        "Thought: Got 30. Now divide by 5.\nAction: divide\nAction Input: {'a': 30, 'b': 5}",
        "Thought: Result is 6.\nAnswer: (10 * 3) / 5 = 6"
    ]
    
    mock_llm = MockLLMWithChain(chain=chain)
    agent = SimpleReActAgent(
        llm=mock_llm,
        tools=[multiply_tool, divide_tool]
    )
    
    handler = agent.run(user_msg="What is (10 * 3) / 5?")
    result = await agent.get_results_from_handler(handler)
    
    assert result["response"] == "(10 * 3) / 5 = 6"
    assert len(result["reasoning"]) == 5  # 2 actions + 2 observations + 1 response
    assert result["sources"] == [30, 6]
```

### 2. Testing Error Recovery

```python
async def test_agent_handles_tool_failure():
    """Test agent recovers from tool errors."""
    
    def failing_tool(x: int) -> str:
        raise ValueError("Tool failed!")
    
    chain = [
        "Thought: Use the tool.\nAction: failing_tool\nAction Input: {'x': 5}",
        "Thought: Tool failed, provide alternative.\nAnswer: I encountered an error but can suggest..."
    ]
    
    mock_llm = MockLLMWithChain(chain=chain)
    tool = FunctionTool.from_defaults(fn=failing_tool)
    
    agent = SimpleReActAgent(llm=mock_llm, tools=[tool])
    handler = agent.run(user_msg="Process this")
    result = await agent.get_results_from_handler(handler)
    
    assert "encountered an error" in result["response"]
```

### 3. Testing Reasoning Patterns

```python
async def test_agent_reasoning_steps():
    """Validate the agent follows expected reasoning patterns."""
    
    chain = [
        "Thought: Analyze the request.\nAction: analyze\nAction Input: {'text': 'input'}",
        "Thought: Process the analysis.\nAction: process\nAction Input: {'data': 'analyzed'}",
        "Thought: Complete.\nAnswer: Processed successfully"
    ]
    
    mock_llm = MockLLMWithChain(chain=chain)
    agent = SimpleReActAgent(llm=mock_llm, tools=[analyze_tool, process_tool])
    
    handler = agent.run(user_msg="Analyze and process this")
    result = await agent.get_results_from_handler(handler)
    
    # Validate reasoning sequence
    reasoning = result["reasoning"]
    assert reasoning[0].action == "analyze"
    assert reasoning[2].action == "process"
    assert isinstance(reasoning[4], ResponseReasoningStep)
```

## Production Patterns

### Environment-Based Configuration

```python
import os
from fm_app_toolkit.agents.llamaindex import SimpleReActAgent

def create_agent(environment: str = None):
    """Create agent based on environment."""
    
    env = environment or os.getenv("ENVIRONMENT", "development")
    
    if env == "development":
        # Use mocks for development
        from fm_app_toolkit.testing import MockLLMWithChain
        llm = MockLLMWithChain(chain=predetermined_responses)
    elif env == "staging":
        # Use cheaper model for staging
        from llama_index.llms.openai import OpenAI
        llm = OpenAI(model="gpt-3.5-turbo")
    else:  # production
        # Use best model for production
        from llama_index.llms.anthropic import Anthropic
        llm = Anthropic(model="claude-3-opus-20240229")
    
    return SimpleReActAgent(
        llm=llm,
        tools=production_tools,
        max_iterations=10
    )
```

### Monitoring and Observability

```python
from fm_app_toolkit.agents.llamaindex import SimpleReActAgent
import structlog

logger = structlog.get_logger()

class MonitoredAgent(SimpleReActAgent):
    """Agent with built-in monitoring."""
    
    async def run(self, user_msg: str, **kwargs):
        """Run with monitoring."""
        start_time = time.time()
        
        try:
            handler = await super().run(user_msg, **kwargs)
            result = await self.get_results_from_handler(handler)
            
            # Log metrics
            logger.info(
                "agent_execution",
                duration=time.time() - start_time,
                num_steps=len(result["reasoning"]),
                num_tool_calls=len(result["sources"]),
                success=True
            )
            
            return handler
        except Exception as e:
            logger.error(
                "agent_execution_failed",
                duration=time.time() - start_time,
                error=str(e)
            )
            raise
```

### Error Recovery Strategies

```python
class ResilientAgent(SimpleReActAgent):
    """Agent with automatic retry and fallback."""
    
    async def run_with_retry(self, user_msg: str, max_retries: int = 3):
        """Run with automatic retry on failure."""
        
        for attempt in range(max_retries):
            try:
                handler = await self.run(user_msg)
                return await self.get_results_from_handler(handler)
            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt failed, use fallback
                    return {
                        "response": "I encountered an issue processing your request. Please try again.",
                        "sources": [],
                        "reasoning": [],
                        "error": str(e)
                    }
                # Wait before retry
                await asyncio.sleep(2 ** attempt)
```

## Creating Custom Tools

### Basic Tool Creation

```python
from llama_index.core.tools import FunctionTool

def get_weather(city: str) -> str:
    """Get weather for a city.
    
    Args:
        city: Name of the city
    
    Returns:
        Weather description
    """
    # Implementation
    return f"Sunny and 72°F in {city}"

weather_tool = FunctionTool.from_defaults(
    fn=get_weather,
    name="get_weather",
    description="Get current weather for a city"
)
```

### Async Tools

```python
async def fetch_data(url: str) -> str:
    """Async tool for fetching data."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async_tool = FunctionTool.from_defaults(
    async_fn=fetch_data,
    name="fetch_data",
    description="Fetch data from a URL"
)
```

### Tools with Complex Schemas

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    """Search parameters."""
    query: str = Field(description="Search query")
    limit: int = Field(default=10, description="Max results")
    filters: dict = Field(default={}, description="Optional filters")

def advanced_search(params: SearchParams) -> str:
    """Perform advanced search."""
    return f"Found {params.limit} results for '{params.query}'"

search_tool = FunctionTool.from_defaults(
    fn=advanced_search,
    name="advanced_search",
    description="Perform advanced search with filters"
)
```

## Debugging Tips

### 1. Enable Verbose Mode

```python
agent = SimpleReActAgent(llm=llm, tools=tools, verbose=True)
# Shows all reasoning steps and tool calls
```

### 2. Inspect Reasoning Steps

```python
result = await agent.get_results_from_handler(handler)

for i, step in enumerate(result["reasoning"]):
    print(f"Step {i}: {type(step).__name__}")
    if hasattr(step, 'action'):
        print(f"  Action: {step.action}")
    if hasattr(step, 'observation'):
        print(f"  Observation: {step.observation}")
```

### 3. Validate ReAct Format

```python
from llama_index.core.agent.react.output_parser import ReActOutputParser

parser = ReActOutputParser()

# Test your LLM outputs
test_response = "Thought: Test.\nAction: tool\nAction Input: {}"
try:
    parsed = parser.parse(test_response)
    print("Valid ReAct format")
except:
    print("Invalid format")
```

## Common Issues and Solutions

### Issue: Agent Doesn't Use Tools

**Cause**: Prompt not clear about available tools
**Solution**: Ensure tools have clear descriptions

```python
tool = FunctionTool.from_defaults(
    fn=my_function,
    description="Use this to calculate X when user asks about Y"  # Be specific
)
```

### Issue: Agent Loops Infinitely

**Cause**: No clear completion condition
**Solution**: Set max_iterations and ensure Answer format is clear

```python
agent = SimpleReActAgent(
    llm=llm,
    tools=tools,
    max_iterations=5  # Prevent infinite loops
)
```

### Issue: Tool Outputs Not Captured

**Cause**: Tool returns wrong type
**Solution**: Ensure tools return strings

```python
def my_tool(x: int) -> str:  # Always return str
    result = x * 2
    return str(result)  # Convert to string
```

## Integration with LlamaIndex Ecosystem

Our SimpleReActAgent is fully compatible with LlamaIndex:

```python
# Works with any LlamaIndex LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.ollama import Ollama

# Works with any LlamaIndex tool
from llama_index.tools import QueryEngineTool, RetrieverTool

# Extends BaseWorkflowAgent for compatibility
from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
```

## Best Practices

1. **Keep Tools Focused**: Each tool should do one thing well
2. **Clear Descriptions**: Tools need clear, specific descriptions
3. **Handle Errors Gracefully**: Tools should return error messages, not raise exceptions
4. **Test with Mocks First**: Always test with MockLLMWithChain before using real LLMs
5. **Monitor in Production**: Log reasoning steps and tool usage for debugging
6. **Set Reasonable Limits**: Use max_iterations to prevent runaway agents
7. **Validate Inputs**: Use Pydantic models for complex tool inputs

## Summary

This module provides a production-ready implementation of the ReAct pattern that:
- Shows exactly what agents look like in code
- Integrates seamlessly with your business logic through tools
- Supports deterministic testing with mock LLMs
- Follows LlamaIndex best practices with BaseWorkflowAgent
- Provides transparency through accessible reasoning steps

Whether you're building a simple calculator or a complex information retrieval system, this agent pattern gives you the foundation for reliable, testable, and observable AI applications.