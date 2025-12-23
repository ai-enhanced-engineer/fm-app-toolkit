# Testing AI Systems - Custom Abstractions for Deterministic Testing

> **Related Article**: [Production AI Systems: The Unit Testing Paradox](https://aienhancedengineer.substack.com/p/production-ai-systems-the-unit-testing)

The unit testing paradox in AI systems: traditional testing breaks down when your system calls non-deterministic language models. Teams face an impossible choice between expensive API-based tests or no automated testing at all.

The breakthrough isn't about replacing LLMs—it's about controlling the interface.

## The Testing Challenge

AI-powered applications introduce unique testing challenges:

- **Cost**: Each test consumes expensive API tokens
- **Latency**: Network calls slow down test suites 
- **Non-determinism**: Models produce different responses each time
- **CI/CD**: API keys shouldn't be required in build environments

Our solution: custom testing abstractions that provide deterministic, controlled test environments without external API calls.

## Custom Testing Abstractions

This module provides mock implementations that extend base LLM classes, enabling drop-in replacement during testing while maintaining full compatibility with production code.

### TrajectoryMockLLMLlamaIndex

Returns responses from a predefined sequence, perfect for testing multi-step agent workflows.

```python
from src.mocks import TrajectoryMockLLMLlamaIndex

# Define exact sequence of responses
mock_llm = TrajectoryMockLLMLlamaIndex(chain=[
    "Thought: I need to search.\nAction: search\nAction Input: {'query': 'python'}",
    "Thought: Found results.\nAnswer: Python is a programming language."
])

# Use with any agent framework
agent = YourAgent(llm=mock_llm, tools=[...])
result = agent.run("Tell me about Python")

# Each LLM call returns the next response in sequence
# First call → search action
# Second call → final answer
```

**Key Features:**
- Sequential response control
- Automatic advancement through chain
- Reset capability for test isolation
- Empty responses when chain exhausted

### MockLLMEchoStream  

Echoes user input back, useful for testing streaming behavior and message flow.

```python
from src.mocks import MockLLMEchoStream

mock_llm = MockLLMEchoStream()

# Non-streaming
response = mock_llm.chat([ChatMessage(role="user", content="Hello world")])
assert response.message.content == "Hello world"

# Streaming (chunks by 7 characters)
stream = mock_llm.stream_chat([ChatMessage(role="user", content="Test message")])
chunks = [chunk.delta for chunk in stream]
assert chunks == ["Test me", "ssage"]
```

### RuleBasedMockLLM

Responds dynamically based on content patterns, offering more flexibility than predefined chains.

```python
from src.mocks import RuleBasedMockLLM

# Define behavior rules
rules = {
    "calculate": "Thought: I'll do the math.\nAction: calculator\nAction Input: {...}",
    "weather": "Thought: I'll check weather.\nAction: get_weather\nAction Input: {...}",
    "time": "Thought: I'll get the time.\nAction: get_current_time\nAction Input: {}"
}

mock_llm = RuleBasedMockLLM(rules=rules, default_behavior="direct_answer")

# Intelligently responds based on query content
agent = YourAgent(llm=mock_llm, tools=[...])
result = agent.run("What's the weather like?")  # Triggers weather rule
```

## Core Testing Patterns

### Sequential Workflow Testing

Test complex agent reasoning with predetermined response chains:

```python
def test_multi_step_calculation():
    chain = [
        "Thought: Multiply first.\nAction: multiply\nAction Input: {'a': 5, 'b': 3}",
        "Thought: Got 15. Add 7.\nAction: add\nAction Input: {'a': 15, 'b': 7}", 
        "Thought: Result is 22.\nAnswer: 5 × 3 + 7 = 22"
    ]
    
    agent = create_agent(TrajectoryMockLLMLlamaIndex(chain=chain))
    result = agent.run("Calculate 5 * 3 + 7")
    
    assert result.response == "5 × 3 + 7 = 22"
    assert result.tool_outputs == [15, 22]  # Verify tool execution sequence
```

### Error Handling Validation

Test how agents handle parsing errors or invalid responses:

```python
def test_agent_error_recovery():
    chain = [
        "Invalid response format",  # Causes parser error
        "Thought: Retry with valid format.\nAnswer: Recovered successfully"
    ]
    
    agent = create_agent(TrajectoryMockLLMLlamaIndex(chain=chain))
    result = agent.run("Test error handling")
    
    assert "Recovered successfully" in result.response
```

### Streaming Behavior Testing

Validate progressive content building in streaming responses:

```python
def test_streaming_content_building():
    mock_llm = MockLLMEchoStream()
    test_content = "Streaming test message"
    
    cumulative_content = ""
    stream = mock_llm.stream_chat([ChatMessage(role="user", content=test_content)])
    
    for chunk in stream:
        cumulative_content += chunk.delta
        assert chunk.message.content == cumulative_content  # Verify progressive building
    
    assert cumulative_content == test_content
```

### Parametrized Edge Cases

Test various scenarios systematically:

```python
@pytest.mark.parametrize("content,expected_chunks", [
    ("", []),                           # Empty content
    ("Hi", ["Hi"]),                     # Less than chunk size  
    ("1234567", ["1234567"]),           # Exactly chunk size
    ("A" * 20, ["A" * 7, "A" * 7, "A" * 6]),  # Multiple chunks
])
def test_streaming_edge_cases(content, expected_chunks):
    mock_llm = MockLLMEchoStream()
    stream = mock_llm.stream_chat([ChatMessage(role="user", content=content)])
    
    chunks = [chunk.delta for chunk in stream]
    assert chunks == expected_chunks
```

## Implementation Examples

### Agent Testing with Controlled Responses

```python
@pytest.fixture
def mock_calculation_agent():
    """Agent that performs multi-step calculations."""
    chain = [
        "Thought: First, multiply.\nAction: multiply\nAction Input: {'a': 2, 'b': 3}",
        "Thought: Got 6. Now add.\nAction: add\nAction Input: {'a': 6, 'b': 4}", 
        "Thought: Result is 10.\nAnswer: 2 × 3 + 4 = 10"
    ]
    
    mock_llm = TrajectoryMockLLMLlamaIndex(chain=chain)
    return YourReActAgent(llm=mock_llm, tools=[multiply_tool, add_tool])

def test_calculation_workflow(mock_calculation_agent):
    result = mock_calculation_agent.run("Calculate 2 * 3 + 4")
    
    assert result.response == "2 × 3 + 4 = 10"
    assert result.tool_outputs == [6, 10]
    assert len(result.reasoning_steps) == 5  # 2 actions + 2 observations + 1 response
```

### Integration with Any Framework

All mocks extend base LLM classes, making them compatible with any framework:

```python
# Works with any LLM-expecting component
mock_llm = TrajectoryMockLLMLlamaIndex(chain=[...])

# Your framework here
agent = create_agent(llm=mock_llm)
query_engine = create_query_engine(llm=mock_llm)  
chat_engine = create_chat_engine(llm=mock_llm)
```

## Best Practices

### Match Expected Formats
Ensure mock responses match your agent's expected format:

```python
# ✅ Good - proper format for ReAct agents
"Thought: I need to search.\nAction: search\nAction Input: {'q': 'test'}"

# ❌ Bad - missing required structure  
"I need to search for test"
```

### Test Both Success and Failure Paths

```python
@pytest.mark.parametrize("chain,expected", [
    (["Thought: Clear.\nAnswer: Success"], "Success"),      # Success case
    (["Invalid format", ""], ""),                          # Failure case
])
def test_various_scenarios(chain, expected):
    agent = create_agent(TrajectoryMockLLMLlamaIndex(chain=chain))
    result = agent.run("Test query") 
    assert expected in result.response
```

### Use Fixtures for Reusability

```python
@pytest.fixture
def mock_llm_factory():
    """Factory for creating mocks with different chains."""
    def _create(chain):
        return TrajectoryMockLLMLlamaIndex(chain=chain)
    return _create

@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset mocks between tests."""
    yield
    # Reset any shared mock state
```

### Helper Functions for Complex Patterns

```python
def generate_expected_chunks(content: str, chunk_size: int = 7) -> list[str]:
    """Generate expected chunks for streaming validation."""
    if not content:
        return []
    return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

def create_react_chain(steps: list[tuple[str, str]]) -> list[str]:
    """Create ReAct format chain from (thought, action) pairs."""
    chain = []
    for i, (thought, action) in enumerate(steps):
        if i == len(steps) - 1:  # Last step is answer
            chain.append(f"Thought: {thought}\nAnswer: {action}")
        else:
            chain.append(f"Thought: {thought}\nAction: {action}\nAction Input: {{}}")
    return chain
```

---

This testing approach transforms AI system testing from an expensive, unpredictable process into a reliable, efficient workflow. By controlling the interface rather than replacing the models, you maintain realistic testing while gaining the determinism necessary for robust test suites.

The key insight: your business logic should be testable independently of the specific LLM responses, enabling you to focus on what your system *does* with those responses rather than what the model *says*.