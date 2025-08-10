# Testing Module - Mock LLMs for LlamaIndex

This module provides mock LLM implementations for testing LlamaIndex applications without making real API calls.

## Overview

Testing LLM-powered applications presents unique challenges:
- **Cost**: Real API calls are expensive during testing
- **Speed**: Network calls slow down test execution
- **Determinism**: Real LLMs produce variable outputs
- **CI/CD**: API keys shouldn't be required in CI environments

Our mock LLMs solve these problems by simulating LLM behavior with predefined, deterministic responses.

## Module Organization

The testing module is now organized into separate files for clarity:
- `mock_chain.py` - MockLLMWithChain for sequential testing
- `mock_echo.py` - MockLLMEchoStream for streaming tests
- `mock_rule_based.py` - RuleBasedMockLLM for behavior-driven tests
- `mocks.py` - Backward compatibility re-exports

## Available Mocks

### MockLLMWithChain

A mock that returns responses from a predefined sequence, perfect for testing multi-step agent workflows.

**Key Features:**
- Returns responses in sequential order
- Each call advances through the chain
- Returns empty responses when exhausted
- Can be reset to replay from the beginning

**How It Works:**
1. Initialize with a list of response strings
2. Each call to `chat()` or `stream_chat()` returns the next response
3. Internal index tracks position in the chain
4. When chain is exhausted, returns empty responses
5. Call `reset()` to start over from the beginning

**Example Usage:**
```python
from ai_test_lab.testing.mock_chain import MockLLMWithChain
# Or: from ai_test_lab.testing.mocks import MockLLMWithChain

# Define the sequence of LLM responses
mock_llm = MockLLMWithChain(chain=[
    "Thought: I need to search for information.\nAction: search\nAction Input: {'query': 'python'}",
    "Thought: Found results. Now I'll summarize.\nAnswer: Python is a programming language."
])

# Use with an agent (BaseWorkflowAgent pattern)
agent = SimpleReActAgent(llm=mock_llm, tools=[...])
handler = agent.run(user_msg="Tell me about Python")
result = await agent.get_results_from_handler(handler)

# The agent will receive responses in order:
# First LLM call -> First chain element (search action)
# Second LLM call -> Second chain element (final answer)
```

### MockLLMEchoStream

A mock that echoes the user's input back, useful for testing streaming behavior.

**Key Features:**
- Echoes the most recent user message
- Streams response in configurable chunk sizes
- Useful for testing streaming handlers
- No state management needed

**How It Works:**
1. Extracts the last user message from input
2. Returns that message as the response
3. In streaming mode, chunks the response by CHUNK_SIZE (default 7)
4. Each chunk includes both delta and cumulative content

**Example Usage:**
```python
from ai_test_lab.testing.mock_echo import MockLLMEchoStream
# Or: from ai_test_lab.testing.mocks import MockLLMEchoStream

mock_llm = MockLLMEchoStream()

# Simple chat
response = mock_llm.chat([
    ChatMessage(role=MessageRole.USER, content="Hello world")
])
assert response.message.content == "Hello world"

# Streaming
stream = mock_llm.stream_chat([
    ChatMessage(role=MessageRole.USER, content="Test message")
])
for chunk in stream:
    print(chunk.delta)  # Prints: "Test me", "ssage"
```

### RuleBasedMockLLM

A mock that dynamically generates responses based on configurable rules, offering more flexibility than predefined chains.

**Key Features:**
- Responds based on content patterns in queries
- Configurable rules for different scenarios
- Default behavior for unmatched patterns
- More maintainable than long predefined chains

**How It Works:**
1. Define rules as pattern-response mappings
2. Mock checks query content against patterns
3. Returns appropriate response based on matched rule
4. Falls back to default behavior if no match

**Example Usage:**
```python
from ai_test_lab.testing.mock_rule_based import RuleBasedMockLLM
# Or: from ai_test_lab.testing.mocks import RuleBasedMockLLM

# Define behavior rules
rules = {
    "calculate": "Thought: I need to perform a calculation.\nAction: calculate\nAction Input: {...}",
    "weather": "Thought: I'll check the weather.\nAction: get_weather\nAction Input: {...}",
    "time": "Thought: I'll get the current time.\nAction: get_current_time\nAction Input: {}"
}

mock_llm = RuleBasedMockLLM(rules=rules, default_behavior="direct_answer")

# The mock intelligently responds based on query content
agent = SimpleReActAgent(llm=mock_llm, tools=[...])
handler = agent.run(user_msg="What's the weather like?")  # Triggers weather rule
result = await agent.get_results_from_handler(handler)
```

## Testing Patterns

### 1. Testing Multi-Step Agent Workflows

Use `MockLLMWithChain` to test complex agent reasoning:

```python
@pytest.fixture
def mock_calculation_agent():
    """Agent that performs multi-step calculations."""
    chain = [
        "Thought: First, multiply 5 by 3.\nAction: multiply\nAction Input: {'a': 5, 'b': 3}",
        "Thought: Got 15. Now add 7.\nAction: add\nAction Input: {'a': 15, 'b': 7}",
        "Thought: Result is 22.\nAnswer: 5 × 3 + 7 = 22"
    ]
    
    mock_llm = MockLLMWithChain(chain=chain)
    return SimpleReActAgent(
        llm=mock_llm,
        tools=[multiply_tool, add_tool]
    )

async def test_multi_step_calculation(mock_calculation_agent):
    handler = mock_calculation_agent.run(user_msg="Calculate 5 * 3 + 7")
    result = await mock_calculation_agent.get_results_from_handler(handler)
    assert result["response"] == "5 × 3 + 7 = 22"
    assert result["sources"] == [15, 22]  # Tool outputs
```

### 2. Testing Error Handling

Test how your agent handles parsing errors or invalid responses:

```python
async def test_agent_handles_invalid_response():
    chain = [
        "This is not a valid ReAct format",  # Will cause parser error
        "Thought: Retry with valid format.\nAnswer: Recovered from error"
    ]
    
    mock_llm = MockLLMWithChain(chain=chain)
    agent = SimpleReActAgent(llm=mock_llm)
    
    handler = agent.run(user_msg="Test error recovery")
    result = await agent.get_results_from_handler(handler)
    assert "Recovered from error" in result["response"]
```

### 3. Testing Streaming Behavior

Use `MockLLMEchoStream` to test streaming handlers:

```python
async def test_streaming_handler():
    mock_llm = MockLLMEchoStream()
    chunks_received = []
    
    async def handle_stream(stream):
        async for chunk in stream:
            chunks_received.append(chunk.delta)
    
    stream = await mock_llm.astream_chat([
        ChatMessage(role=MessageRole.USER, content="Test streaming")
    ])
    await handle_stream(stream)
    
    assert "".join(chunks_received) == "Test streaming"
    assert len(chunks_received) == 2  # "Test st" + "reaming"
```

### 4. Testing Chain Exhaustion

Test how your application handles when the mock runs out of responses:

```python
def test_chain_exhaustion():
    mock_llm = MockLLMWithChain(chain=["Response 1", "Response 2"])
    
    # Use up the chain
    response1 = mock_llm.chat([...])
    response2 = mock_llm.chat([...])
    
    # Third call returns empty
    response3 = mock_llm.chat([...])
    assert response3.message.content == ""
    
    # Reset to reuse
    mock_llm.reset()
    response4 = mock_llm.chat([...])
    assert response4.message.content == "Response 1"
```

### 5. Testing Cumulative Content Building

Validate that streaming builds content correctly:

```python
def test_mock_llm_echo_stream_cumulative_content_building():
    test_content = "Tell me about the universe."
    messages = [ChatMessage(role=MessageRole.USER, content=test_content)]
    
    expected_chunks = generate_expected_chunks(test_content)
    cumulative_content = ""
    stream = mock_llm_echo.stream_chat(messages)
    
    for i, response in enumerate(stream):
        # Verify delta matches expected chunk
        assert response.delta == expected_chunks[i]
        
        # Build and verify cumulative content
        cumulative_content += response.delta
        assert response.message.content == cumulative_content
        assert response.message.role == MessageRole.ASSISTANT
    
    # Verify final content matches original
    assert cumulative_content == test_content
```

### 6. Testing Agent Integration Sequence

Validate MockLLMWithChain returns responses in exact order when used by an agent:

```python
async def test_mock_llm_chain_agent_integration_sequence():
    # Define exact sequence of LLM responses
    expected_chain = [
        "Thought: I need to multiply first.\nAction: multiply\nAction Input: {'a': 2, 'b': 3}",
        "Thought: Got 6. Now add 4.\nAction: add\nAction Input: {'a': 6, 'b': 4}",  
        "Thought: Result is 10.\nAnswer: 2 × 3 + 4 = 10"
    ]
    
    agent = SimpleReActAgent(
        llm=MockLLMWithChain(chain=expected_chain),
        tools=[multiply_tool, add_tool]
    )
    
    # Run agent with WorkflowHandler pattern
    handler = agent.run(user_msg="Calculate 2 * 3 + 4")
    result = await agent.get_results_from_handler(handler)
    
    # Validate the agent processed all responses in order
    assert result["response"] == "2 × 3 + 4 = 10"
    assert result["sources"] == [6, 10]  # Tool outputs in sequence
    
    # Validate reasoning steps match expected sequence
    assert result["reasoning"][0].action == "multiply"
    assert result["reasoning"][2].action == "add"
```

### 7. Testing Various Message Lengths

Use parametrized tests for comprehensive edge case coverage:

```python
@pytest.mark.parametrize("content,expected_chunks", [
    ("", []),  # Empty content
    ("Hi", ["Hi"]),  # Less than chunk size
    ("1234567", ["1234567"]),  # Exactly chunk size (7)
    ("12345678", ["1234567", "8"]),  # Slightly over
    ("A" * 20, ["A" * 7, "A" * 7, "A" * 6"]),  # Multiple chunks
])
def test_mock_llm_echo_stream_various_lengths(
    mock_llm_echo: MockLLMEchoStream, 
    content: str, 
    expected_chunks: list[str]
) -> None:
    messages = [ChatMessage(role=MessageRole.USER, content=content)]
    stream = mock_llm_echo.stream_chat(messages)
    
    chunks = [chunk.delta for chunk in stream]
    assert chunks == expected_chunks
```

## Best Practices

### 1. Match ReAct Format Exactly
When testing ReAct agents, ensure your mock responses match the expected format:
```python
# Good - proper ReAct format
"Thought: I need to search.\nAction: search\nAction Input: {'q': 'test'}"

# Bad - missing required fields
"I need to search for test"
```

### 2. Test Both Success and Failure Paths
```python
@pytest.mark.parametrize("chain,expected", [
    # Success case
    (["Thought: Clear.\nAnswer: Success"], "Success"),
    # Failure case  
    (["Invalid format", ""], ""),
])
def test_various_scenarios(chain, expected):
    mock_llm = MockLLMWithChain(chain=chain)
    # ... test implementation
```

### 3. Use Fixtures for Reusability
```python
@pytest.fixture
def mock_llm_factory():
    """Factory for creating mock LLMs with different chains."""
    def _create(chain):
        return MockLLMWithChain(chain=chain)
    return _create
```

### 4. Reset Between Tests
```python
@pytest.fixture
def mock_llm():
    llm = MockLLMWithChain(chain=["Response 1", "Response 2"])
    yield llm
    llm.reset()  # Clean up for next test
```

### 5. Use Helper Functions for Complex Patterns
```python
def generate_expected_chunks(content: str, chunk_size: int = 7) -> list[str]:
    """Generate expected chunks from content for streaming validation."""
    if not content:
        return []
    return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
```

### 6. Validate Streaming at Each Step
```python
def test_streaming():
    cumulative = ""
    for chunk in stream:
        cumulative += chunk.delta
        # Validate cumulative content at each step
        assert chunk.message.content == cumulative
```

## WorkflowHandler Pattern (Production-Grade)

SimpleReActAgent now follows production patterns by returning a WorkflowHandler:

```python
# Production-grade pattern with WorkflowHandler
handler = agent.run(user_msg="What is 5 + 3?")

# Extract results using helper method
result = await agent.get_results_from_handler(handler)

# Access extracted data
print(result["response"])   # The agent's answer
print(result["sources"])    # Tool outputs used
print(result["reasoning"])  # Reasoning steps taken
print(result["chat_history"])  # Conversation history
```

This pattern:
- Matches how production LlamaIndex agents work
- Provides access to streaming events if needed
- Maintains compatibility with the LlamaIndex ecosystem
- Shows clear separation between execution and result extraction

## Integration with LlamaIndex

All mocks extend `llama_index.core.llms.llm.LLM`, making them drop-in replacements for real LLMs:

```python
from llama_index.core.agent import ReActAgent
from ai_test_lab.testing.mocks import MockLLMWithChain

# Works with any LlamaIndex component expecting an LLM
mock_llm = MockLLMWithChain(chain=[...])

# ReActAgent
agent = ReActAgent.from_tools(tools, llm=mock_llm)

# Query engines
query_engine = index.as_query_engine(llm=mock_llm)

# Chat engines  
chat_engine = index.as_chat_engine(llm=mock_llm)
```

## Debugging Tips

### 1. Verbose Mode
Enable verbose logging to see the exact LLM calls:
```python
agent = SimpleReActAgent(llm=mock_llm, verbose=True)
```

### 2. Track Chain Progress
```python
mock_llm = MockLLMWithChain(chain=[...])
print(f"Current index: {mock_llm._current_index}")
print(f"Chain length: {len(mock_llm.message_chain)}")
```

### 3. Validate Response Format
```python
from llama_index.core.agent.react.output_parser import ReActOutputParser

parser = ReActOutputParser()
for response in chain:
    try:
        parser.parse(response)
        print(f"✓ Valid: {response[:50]}...")
    except:
        print(f"✗ Invalid: {response[:50]}...")
```

## Common Pitfalls

1. **Forgetting to reset**: Reusing a mock without resetting will continue from where it left off
2. **Mismatched formats**: Ensure mock responses match your agent's expected format
3. **Index out of bounds**: Chain exhaustion returns empty responses, not errors
4. **Streaming vs non-streaming**: Both methods advance the index, don't mix them unintentionally
5. **Empty content handling**: Empty strings produce no chunks in streaming, not a single empty chunk
6. **Cumulative content validation**: Always verify both delta and cumulative content in streaming tests

## Summary

These mock LLMs enable fast, deterministic, and cost-free testing of LlamaIndex applications. By simulating LLM behavior with predefined responses, you can thoroughly test your agent logic, error handling, and multi-step workflows without external dependencies.