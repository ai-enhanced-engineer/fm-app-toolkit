# LLM Application Testing Strategies

A demonstration project showing how to effectively test LLM-powered applications, specifically LlamaIndex agent implementations, using mock LLMs for deterministic, fast, and cost-effective testing.

## What is this?

This project demonstrates a powerful testing pattern for LLM applications that allows you to:
- Write deterministic unit tests without calling actual LLM APIs
- Test complex agent behaviors and tool interactions
- Run tests in CI/CD without API keys
- Debug agent logic step-by-step
- Save costs during development and testing

The key innovation is creating mock LLM implementations that extend LlamaIndex's base LLM class, enabling full integration testing of ReActAgent workflows without network dependencies.

## 🎯 Key Features

- **Mock LLM Implementations**: Two mock classes that extend LlamaIndex's base LLM
  - `MockLLMWithChain`: Returns predefined response sequences
  - `MockLLMEchoStream`: Echoes input for testing streaming
- **Multiple Agent Implementations**: 
  - Standard LlamaIndex `ReActAgent` with mocks
  - Custom `SimpleReActAgent` using Workflow pattern for pedagogical clarity
- **Deterministic Testing**: Control exact LLM responses for predictable tests
- **Zero Network Calls**: Tests run offline without API dependencies
- **Comprehensive Examples**: 30+ tests demonstrating various patterns

## Quick Start

### Prerequisites
- Python 3.12+
- Make

### Setup

1. Clone the repository:
```bash
git clone <repository-url> ai-test-lab
cd ai-test-lab
```

2. Create environment and install dependencies:
```bash
make environment-create
```

3. Run the tests to see the patterns in action:
```bash
make unit-test
```

## The Testing Pattern

```python
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from ai_test_lab.testing.mocks import MockLLMWithChain
from ai_test_lab.tools import add

# 1. Create mock with predefined ReAct-formatted responses
mock_llm = MockLLMWithChain(chain=[
    "Thought: I need to add these numbers\nAction: add\nAction Input: {'a': 5, 'b': 3}",
    "Thought: I have the result\nAnswer: The sum is 8"
])

# 2. Create agent with mock LLM
agent = ReActAgent(
    tools=[FunctionTool.from_defaults(fn=add)],
    llm=mock_llm
)

# 3. Test deterministically
response = await agent.run(user_msg="What is 5 + 3?")
assert "8" in str(response)
```

## Project Structure

```
ai-test-lab/
├── ai_test_lab/
│   ├── agents/
│   │   ├── simple_react.py  # Pedagogical ReAct agent using Workflow
│   │   └── events.py        # Workflow events for agent communication
│   ├── testing/
│   │   ├── __init__.py
│   │   └── mocks.py         # Mock LLM implementations
│   ├── tools.py             # Example tools for testing
│   └── main.py              # FastAPI entry point (optional)
├── tests/
│   ├── test_mock_llms.py    # Tests for mock implementations
│   ├── test_react_agent_with_mocks.py  # ReActAgent integration tests
│   └── test_simple_react_agent.py      # SimpleReActAgent tests
├── Makefile                 # Development commands
├── pyproject.toml           # Project configuration
├── CLAUDE.md                # Project-specific development guide
├── ADR.md                   # Architecture Decision Record
└── README.md                # This documentation
```

## Agent Implementations

### SimpleReActAgent
A pedagogical implementation that clearly demonstrates the ReAct pattern using LlamaIndex's Workflow architecture:

```python
from ai_test_lab.agents.simple_react import SimpleReActAgent
from ai_test_lab.testing.mocks import MockLLMWithChain

# Create agent with mock LLM
agent = SimpleReActAgent(
    tools=tools,
    llm=MockLLMWithChain(chain=["..."]),
    verbose=True  # Shows reasoning steps
)

# Run the agent
result = await agent.run(user_query="What is 5 + 3?")
```

### Standard ReActAgent
You can also test the standard LlamaIndex ReActAgent with our mocks:

```python
from llama_index.core.agent.workflow import ReActAgent

agent = ReActAgent(
    tools=tools,
    llm=mock_llm
)
```

## Mock LLM Implementations

### MockLLMWithChain
Returns a predefined sequence of responses. Perfect for testing ReAct agent reasoning chains:

```python
mock_llm = MockLLMWithChain(chain=[
    "First response",
    "Second response",
    "Third response"
])
```

### MockLLMEchoStream
Echoes user input back in streaming chunks. Useful for testing streaming behavior:

```python
mock_llm = MockLLMEchoStream()
# Will echo whatever the user sends, in chunks
```

## Testing Patterns Demonstrated

### ✅ Basic Tool Usage
Test that agents correctly use tools with expected inputs and handle outputs.

### ✅ Multi-Step Reasoning
Verify complex reasoning chains with multiple tool calls.

### ✅ Error Handling
Test how agents handle tool failures and edge cases.

### ✅ Tool Selection
Ensure agents select the right tool from multiple available options.

### ✅ Direct Responses
Test scenarios where agents answer without using tools.

### ✅ Response Processing
Verify response parsing and formatting.

### ✅ Workflow Events
Test event-driven communication between workflow steps in the SimpleReActAgent.

## Development Workflow

### Essential Commands

```bash
# Environment
make environment-create   # First-time setup
make environment-sync     # Update dependencies

# Testing
make unit-test           # Run all tests
make validate-branch     # Run linting and tests

# Code Quality
make format              # Auto-format code
make lint               # Fix linting issues
```

### Adding Your Own Tests

1. Create mock response chains matching ReAct format
2. Initialize ReActAgent with mock LLM and tools
3. Run agent with test inputs
4. Assert expected behaviors

## Benefits of This Approach

### 🚀 Fast Development
- No waiting for API responses
- Instant test execution
- Rapid iteration on agent logic

### 💰 Cost-Effective
- Zero API costs during testing
- Unlimited test runs
- No rate limiting concerns

### 🎯 Reliable Testing
- 100% deterministic results
- Reproducible test failures
- Easy debugging with controlled responses

### 🔧 CI/CD Ready
- No API keys needed in CI
- Tests run in isolated environments
- Consistent results across runs

## Best Practices

1. **Be Specific with Responses**: Format your mock responses exactly as the ReAct agent expects them
2. **Test Edge Cases**: Use mocks to test error conditions that are hard to reproduce with real LLMs
3. **Reset Between Tests**: Call `mock_llm.reset()` to replay chains in multiple tests
4. **Combine with Real LLMs**: Use mocks for unit tests, real LLMs for integration tests
5. **Document Response Formats**: Keep examples of expected ReAct formatting for reference

## Advanced Patterns

### Custom Mock Behaviors
Extend the base mocks for specific testing needs:

```python
class MockLLMWithConditionalResponse(MockLLMWithChain):
    def chat(self, messages, **kwargs):
        # Custom logic based on message content
        if "error" in messages[-1].content.lower():
            return ChatResponse(message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Error handling response"
            ))
        return super().chat(messages, **kwargs)
```

### Testing Memory and Context
The mocks preserve conversation context just like real LLMs:

```python
mock_llm = MockLLMWithChain(chain=[
    "Response 1 based on context",
    "Response 2 remembering previous"
])

agent = ReActAgent(tools=[], llm=mock_llm)
response1 = await agent.run(user_msg="First question")
response2 = await agent.run(user_msg="Follow-up question")
```

## When to Use This Pattern

✅ **Perfect for:**
- Unit testing agent logic
- Integration testing tool usage
- Testing error handling
- CI/CD pipelines
- Local development

❌ **Not suitable for:**
- Testing actual LLM performance
- Prompt engineering validation
- Model behavior verification
- Production response quality

## Limitations

- Mock responses must match ReAct format exactly
- Complex tool interactions may require careful response crafting
- Streaming behavior is simplified compared to real LLMs

## Contributing

This pattern can be extended with:
- Additional mock behaviors
- More complex tool examples
- Performance testing utilities
- Test data generators

## Related Documentation

- [Architecture Decision Record](ADR.md) - Technical decisions and rationale
- [LlamaIndex Docs](https://docs.llamaindex.ai/) - Official LlamaIndex documentation

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

---

**Built to demonstrate effective testing strategies for LLM applications** 🧪🤖