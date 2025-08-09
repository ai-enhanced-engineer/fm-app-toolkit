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
├── ai_test_lab/              # Main package
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

### Mock LLMs (`ai_test_lab/testing/mocks.py`)
- **MockLLMWithChain**: Returns predefined response sequences
- **MockLLMEchoStream**: Echoes input for testing streaming behavior
- Both extend LlamaIndex's base LLM class for full compatibility

### Simple ReAct Agent (`ai_test_lab/agents/simple_react.py`)
- Pedagogical implementation of ReAct pattern
- Subclasses from `llama_index.core.workflow.Workflow`
- Uses standard ReActChatFormatter and ReActOutputParser
- Demonstrates the reasoning loop and action calling clearly

### Workflow Events (`ai_test_lab/agents/events.py`)
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

1. **Deterministic LLM Testing**: Control exact LLM responses
2. **Tool Usage Testing**: Verify agents use tools correctly
3. **Multi-Step Reasoning**: Test complex reasoning chains
4. **Error Handling**: Test failure scenarios
5. **Streaming Behavior**: Test async streaming responses
6. **Workflow Integration**: Test event-driven agent workflows

## Best Practices for This Project

### When Adding New Tests
- Use MockLLMWithChain for deterministic response sequences
- Format mock responses to match ReAct format exactly
- Reset mocks between test runs with `mock_llm.reset()`
- Test both success and failure paths

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
uv run python -m pytest tests/test_simple_react_agent.py::TestSimpleReActAgent::test_single_tool_execution -v
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