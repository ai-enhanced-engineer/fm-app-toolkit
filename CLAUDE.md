# LLM Testing Strategies - Development Guide

A demonstration project for testing LLM-powered applications, specifically showcasing how to test LlamaIndex agents using mock LLMs for deterministic, fast, and cost-effective testing.

## Project Overview

This project demonstrates best practices for testing LLM applications by:
- Creating mock LLM implementations that extend LlamaIndex's base classes
- Testing ReAct agents without making actual API calls
- Running deterministic tests with controlled LLM responses
- Implementing both simple pedagogical agents and production-ready testing patterns

## Project Structure

```
ai-test-lab/
├── fm_app_toolkit/              # Main package
│   ├── agents/               # Agent implementations
│   │   ├── simple_react.py  # Pedagogical ReAct agent using Workflow
│   │   └── events.py        # Workflow events for agent communication
│   ├── testing/             # Testing utilities
│   │   └── mocks.py        # Mock LLM implementations
│   ├── tools.py            # Example tools for agents
│   └── main.py             # FastAPI entry point (optional)
├── tests/                   # Comprehensive test suite
│   ├── test_mock_llms.py   # Tests for mock implementations
│   ├── test_react_agent_with_mocks.py  # ReActAgent tests
│   └── test_simple_react_agent.py      # SimpleReActAgent tests
├── research/                # Notebooks and experiments
├── Makefile                # Development automation
└── pyproject.toml          # Project config & dependencies
```

## Key Components

### Mock LLMs (`fm_app_toolkit/testing/mocks.py`)
- **MockLLMWithChain**: Returns predefined response sequences
- **MockLLMEchoStream**: Echoes input for testing streaming behavior
- Both extend LlamaIndex's base LLM class for full compatibility

### Simple ReAct Agent (`fm_app_toolkit/agents/simple_react.py`)
- Pedagogical implementation of ReAct pattern
- Subclasses from `llama_index.core.workflow.Workflow`
- Uses standard ReActChatFormatter and ReActOutputParser
- Demonstrates the reasoning loop and action calling clearly

### Workflow Events (`fm_app_toolkit/agents/events.py`)
- Event-driven communication between workflow steps
- PrepEvent, InputEvent, ToolCallEvent, StopEvent
- Clean separation of concerns in agent workflow

## Development Workflow

### Quick Start
```bash
make environment-create   # Creates Python 3.12 env with uv
make environment-sync     # Updates dependencies after changes
```

### Testing Commands
```bash
make unit-test           # Run all 32+ tests
make validate-branch     # Run linting and tests before PR
make all-test           # Run with coverage report
```

### Code Quality
```bash
make format              # Auto-format with Ruff
make lint               # Lint and auto-fix issues
make type-check         # Type check with MyPy
```

## Testing Patterns Demonstrated

### Core Testing Patterns
1. **Deterministic LLM Testing**: Control exact LLM responses
2. **Tool Usage Testing**: Verify agents use tools correctly
3. **Multi-Step Reasoning**: Test complex reasoning chains
4. **Error Handling**: Test failure scenarios
5. **Streaming Behavior**: Test async streaming responses
6. **Workflow Integration**: Test event-driven agent workflows

### Advanced Testing Patterns

#### Sequential Response Validation
Test that MockLLMWithChain returns responses in exact order:
```python
async def test_mock_llm_chain_agent_integration_sequence():
    chain = [
        "Thought: First step.\nAction: tool1\nAction Input: {...}",
        "Thought: Second step.\nAction: tool2\nAction Input: {...}",
        "Thought: Done.\nAnswer: Final result"
    ]
    mock_llm = MockLLMWithChain(chain=chain)
    # Agent receives responses in exact sequence
```

#### Cumulative Content Validation
Verify streaming builds content correctly:
```python
def test_streaming_cumulative_content():
    for i, chunk in enumerate(stream):
        cumulative_content += chunk.delta
        assert response.message.content == cumulative_content
```

#### Parametrized Edge Case Testing
Test various message lengths and edge cases:
```python
@pytest.mark.parametrize("content,expected_chunks", [
    ("", []),  # Empty content
    ("Hi", ["Hi"]),  # Less than chunk size
    ("1234567", ["1234567"]),  # Exactly chunk size
    ("A" * 20, ["A" * 7, "A" * 7, "A" * 6"]),  # Multiple chunks
])
```

## Best Practices for This Project

### When Adding New Tests
- Use MockLLMWithChain for deterministic response sequences
- Format mock responses to match ReAct format exactly
- Reset mocks between test runs with `mock_llm.reset()`
- Test both success and failure paths
- Use fixtures for mock creation to ensure consistency
- Add helper functions for complex test patterns (e.g., `generate_expected_chunks()`)
- Include parametrized tests for edge cases
- Validate both delta and cumulative content in streaming tests

### When Modifying Agents
- Maintain compatibility with existing mock LLMs
- Keep the pedagogical clarity of SimpleReActAgent
- Ensure all workflow events are properly handled
- Add corresponding tests for new functionality

### Code Conventions
- Type hints on all functions (enforced by MyPy)
- Max line length: 120 characters
- Use `@step` decorators for workflow steps
- Follow existing patterns in the codebase

### Test Organization
- Use flat test structure without classes
- Create reusable fixtures for common setup
- Group related tests logically with comments
- Use descriptive test names that explain what is being tested
- Include docstrings that describe the test scenario

## Dependencies Management

### Core Dependencies
- **llama-index-core**: Base LlamaIndex functionality
- **pydantic**: Data validation
- **fastapi**: Optional web framework

### Development Dependencies
- **pytest**: Testing framework
- **pytest-asyncio**: Async test support
- **mypy**: Type checking
- **ruff**: Linting and formatting

### Adding Dependencies
```bash
uv add <package>           # Add runtime dependency
uv add --dev <package>     # Add development dependency
make environment-sync      # Sync environment
```

## Architecture Decisions

### Why Workflow Pattern?
- Clean separation of agent steps
- Event-driven architecture
- Better testability and debugging
- Standard LlamaIndex pattern

### Why Mock LLMs?
- Zero API costs during testing
- Deterministic test results
- Fast test execution
- CI/CD friendly (no API keys needed)

### Why ReAct Pattern?
- Industry standard for agent reasoning
- Clear thought-action-observation loop
- Easy to understand and debug
- Well-supported by LlamaIndex

## Common Tasks

### Running a Specific Test
```bash
# Run a single test
uv run python -m pytest tests/test_simple_react_agent.py::test_single_tool_execution -v

# Run parametrized tests
uv run python -m pytest tests/test_mock_llms.py::test_mock_llm_echo_stream_various_lengths -v

# Run with specific parameter
uv run python -m pytest tests/test_mock_llms.py::test_mock_llm_echo_stream_various_lengths[Hi-expected_chunks1] -v
```

### Debugging Agent Behavior
The SimpleReActAgent includes verbose logging:
- Set verbose=True when creating the agent
- Observe the reasoning steps and tool calls
- Use mock LLMs to control the exact flow

### Creating New Mock Patterns
Extend MockLLMWithChain for custom behaviors:
```python
mock_llm = MockLLMWithChain(chain=[
    "Thought: I need to...\nAction: tool_name\nAction Input: {...}",
    "Thought: Now I have...\nAnswer: Final response"
])
```

## Project Goals

1. **Educational**: Demonstrate clear patterns for testing LLM applications
2. **Practical**: Provide working code that can be adapted for real projects
3. **Comprehensive**: Show various testing scenarios and edge cases
4. **Maintainable**: Keep code simple and well-documented

## Next Steps for Contributors

- Add more mock LLM behaviors for specific test scenarios
- Implement additional agent patterns (e.g., Plan-and-Execute)
- Create performance benchmarks
- Add integration tests with real LLMs (optional)
- Expand tool library with more examples

This project serves as both a learning resource and a practical template for testing LLM applications effectively.