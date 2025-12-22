# Agent Testing Examples

Hands-on examples for the **[AI Agents in Production: Testing the Reasoning Loop](https://aienhancedengineer.substack.com/p/ai-agents-in-production-testing-the)** article (Part 3: Deterministic Testing with Trajectory Mocking).

> **Getting Started**: Run `make environment-create` from the repository root. See the [main README](../../README.md#-development-workflow) for full setup.

This directory contains working test code that demonstrates the deterministic testing patterns described in the article. Each test maps to specific concepts—use them as templates for your own agent tests.

## Article Concepts → Test Files

| Article Concept | Framework | Test File | What It Demonstrates |
|-----------------|-----------|-----------|---------------------|
| **Reasoning Trajectory Mocking** | LlamaIndex | `llamaindex/test_react_agent_with_mocks.py` | Pre-defined LLM response chains |
| **Multi-Step Reasoning** | LlamaIndex | `test__react_agent__multi_step_reasoning` | Sequential tool calls with mock chains |
| **Error Recovery Testing** | LlamaIndex | `test__react_agent__error_handling` | Graceful degradation on tool errors |
| **Tool Selection Verification** | LangGraph | `langgraph/test_simple_react.py` | Correct tool invocation from multiple options |
| **Source Tracking** | LangGraph | `test__multi_step_reasoning__accumulates_multiple_sources` | Verifying tool call accumulation |
| **Structured Output Validation** | PydanticAI | `pydantic/test_pydantic_agent.py` | Type-safe output with TestModel |

## Module Organization

```
tests/agents/
├── llamaindex/                      # LlamaIndex ReAct agent tests
│   ├── test_react_agent_with_mocks.py   # Main mock chain patterns
│   ├── test_simple_react_agent.py       # Integration patterns
│   └── test_minimal_react.py            # Minimal loop tests
│
├── langgraph/                       # LangGraph agent tests
│   ├── test_simple_react.py             # Tool selection & sources
│   └── test_minimal_react.py            # Graph-based loop tests
│
└── pydantic/                        # PydanticAI agent tests
    └── test_pydantic_agent.py           # Structured output validation
```

## Mock Approaches by Framework

The article describes three framework-specific mocking strategies. Here's how they appear in the tests:

### LlamaIndex: `TrajectoryMockLLMLlamaIndex`

```python
# From test_react_agent_with_mocks.py
mock_llm = TrajectoryMockLLMLlamaIndex(
    chain=[
        "Thought: I need to multiply 4 by 5.\nAction: multiply\nAction Input: {'a': 4, 'b': 5}",
        "Thought: Now add 10.\nAction: add\nAction Input: {'a': 20, 'b': 10}",
        "Thought: Done.\nAnswer: The result is 30.",
    ]
)
```

### LangGraph: `TrajectoryMockLLMLangChain`

```python
# From test_simple_react.py
mock_llm = TrajectoryMockLLMLangChain(
    chain=[
        'Thought: I need weather data.\nAction: get_weather\nAction Input: {"location": "Tokyo"}',
        "Thought: I have the information.\nAnswer: Weather in Tokyo: 75°F and sunny",
    ]
)
```

### PydanticAI: `TestModel`

```python
# From test_pydantic_agent.py
test_model = TestModel(
    custom_output_args={
        "entities": ["Apple", "iPhone"],
        "numbers": [394.3, 2022],
        "sentiment": "positive",
        "confidence": 0.95,
    }
)
```

## Key Testing Patterns Demonstrated

### 1. Happy Path Coverage
Every test file includes basic success scenarios—agent receives query, calls correct tool, returns expected answer.

### 2. Multi-Step Reasoning Chains
Tests verify that agents correctly sequence multiple tool calls:
- `test__react_agent__multi_step_reasoning` (LlamaIndex)
- `test__multi_step_reasoning__accumulates_multiple_sources` (LangGraph)

### 3. Error Injection
Tests validate graceful handling of failures:
- `test__react_agent__error_handling` — Division by zero
- `test__tool_not_found__adds_error_to_sources` — Nonexistent tool

### 4. Tool Selection Accuracy
Tests with multiple available tools verify the agent picks correctly:
- `test__react_agent__tool_selection` (LlamaIndex)
- `test__multiple_tools_available__selects_correct_tool` (LangGraph)

### 5. Direct Answers (No Tool Use)
Tests verify agents can answer without tools when appropriate:
- `test__react_agent__direct_answer` (LlamaIndex)
- `test__direct_answer__returns_response_with_no_sources` (LangGraph)

## Running the Tests

```bash
# All agent tests
pytest tests/agents/ -v

# Single framework
pytest tests/agents/llamaindex/ -v
pytest tests/agents/langgraph/ -v
pytest tests/agents/pydantic/ -v

# Single test file
pytest tests/agents/llamaindex/test_react_agent_with_mocks.py -v
```

## Extending These Tests

Use these patterns as templates:

1. **New tool scenarios**: Copy a test, modify the mock chain to include your tool's expected Action/Action Input
2. **Edge cases**: Add error conditions to the chain and verify graceful handling
3. **Complex workflows**: Chain more steps—the mocks support arbitrary sequence lengths

## Related Resources

- **Article**: [AI Agents in Production: Testing the Reasoning Loop](https://aienhancedengineer.substack.com/p/ai-agents-in-production-testing-the)
- **Mock implementations**: `src/testing/` directory
- **Agent implementations**: `src/agents/` directory
