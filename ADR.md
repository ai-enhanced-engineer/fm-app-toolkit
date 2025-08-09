# ADR-001: LLM Testing Strategy Demonstration

## Status
Accepted

## Context
Demonstrating effective testing strategies for LLM-powered applications, specifically showing how to create deterministic unit tests for LlamaIndex ReActAgent applications without requiring network calls to actual LLM services.

## Decision
Implement mock LLM classes that extend LlamaIndex's base LLM class, enabling developers to write fast, reliable, and cost-effective tests for their LLM applications by controlling exact response sequences.

## Consequences
- **Pros**: 
  - Deterministic testing with predictable outcomes
  - No API costs during test execution
  - Fast test execution without network latency
  - Full control over edge cases and error scenarios
  - Tests can run in CI/CD without API keys
- **Cons**: 
  - Mock responses must be carefully crafted to match ReAct format
  - Doesn't test actual LLM behavior or prompt engineering
  - Requires maintaining mock response chains

## Technical Specification
- **Stack**: Python 3.12, LlamaIndex Core, pytest-asyncio
- **Mock Implementations**: 
  - `MockLLMWithChain`: Returns predefined response sequences
  - `MockLLMEchoStream`: Echoes input for streaming tests
- **Agent Framework**: LlamaIndex workflow-based ReActAgent
- **Testing Framework**: pytest with async support
- **Integration**: Mock LLMs implement all required LlamaIndex LLM interface methods

## Testing Pattern
```python
# 1. Create mock with predefined responses
mock_llm = MockLLMWithChain(chain=[
    "Thought: I need to calculate\nAction: add\nAction Input: {'a': 2, 'b': 3}",
    "Thought: Done\nAnswer: The result is 5"
])

# 2. Initialize ReActAgent with mock
agent = ReActAgent(tools=[add_tool], llm=mock_llm)

# 3. Test deterministically
response = await agent.run(user_msg="What is 2+3?")
assert "5" in str(response)
```

## Integration Points
- **Extends**: `llama_index.core.llms.llm.LLM` base class
- **Compatible With**: LlamaIndex ReActAgent, FunctionTool, workflows
- **Test Coverage**: Unit tests for mocks, integration tests with ReActAgent

## Non-Functional Requirements
- **Performance**: Tests run in milliseconds without network calls
- **Reliability**: 100% deterministic responses
- **Maintainability**: Clear separation between mocks and production code
- **Portability**: No external dependencies beyond LlamaIndex

## Use Cases
- **Unit Testing**: Test agent logic without LLM variability
- **Integration Testing**: Verify tool execution and reasoning chains
- **Error Handling**: Test edge cases and failure scenarios
- **CI/CD**: Run tests without API keys or network access
- **Development**: Rapid iteration without API costs