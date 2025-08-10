# FM App Toolkit

**Foundation Model Application Toolkit** - Concrete implementation and testing patterns for developing production-grade foundation model based applications.

## Mission

This toolkit provides battle-tested patterns for building production FM applications using LlamaIndex as a foundational layer, helping developers bring business logic closer to reality through:

- **Deterministic Testing Strategies** - Mock LLMs for cost-effective, reliable testing
- **Production-Ready Agent Implementations** - ReAct agents with clear reasoning patterns
- **Reusable Patterns** - Common solutions for FM app challenges
- **Cost-Effective Development** - Build and test without API calls
- **Business Logic Integration** - Bridge the gap between FM capabilities and real-world requirements

## 🎯 What Problems Does This Solve?

Building production FM applications presents unique challenges:

1. **Testing Complexity** - How do you test non-deterministic LLM outputs?
2. **Development Costs** - API calls during development add up quickly
3. **Integration Patterns** - How do you connect FM capabilities to business logic?
4. **Production Readiness** - Moving from prototype to production-grade code
5. **Debugging Difficulty** - Understanding agent reasoning and decision-making

This toolkit addresses each of these challenges with practical, reusable solutions.

## 🚀 Key Components

### Mock LLM Framework
Complete mock implementations that extend LlamaIndex's base LLM class:
- **MockLLMWithChain** - Sequential response patterns for multi-step workflows
- **MockLLMEchoStream** - Streaming behavior testing
- **RuleBasedMockLLM** - Dynamic responses based on configurable rules

### Agent Implementations
Production-ready agent patterns:
- **SimpleReActAgent** - Clear implementation of the ReAct pattern using BaseWorkflowAgent
- **WorkflowHandler Pattern** - Production-grade event handling and result extraction
- **Tool Integration** - Seamless connection between agents and business logic

### Testing Patterns
Comprehensive testing strategies:
- Deterministic unit tests without API calls
- Multi-step reasoning validation
- Error handling and edge cases
- Tool selection and usage patterns
- Streaming and async behavior

## Quick Start

### Prerequisites
- Python 3.12+
- Make

### Installation

```bash
# Clone the repository
git clone <repository-url> fm-app-toolkit
cd fm-app-toolkit

# Create environment and install dependencies
make environment-create

# Run tests to verify setup
make unit-test
```

### Basic Usage

```python
from fm_app_toolkit.agents import SimpleReActAgent
from fm_app_toolkit.testing import MockLLMWithChain
from fm_app_toolkit.tools import Tool

# Define your business logic as tools
def calculate_price(quantity: int, unit_price: float) -> float:
    """Calculate total price with business rules."""
    return quantity * unit_price * 0.9  # 10% discount

tool = Tool(
    name="calculate_price",
    function=calculate_price,
    description="Calculate price with discount"
)

# Create agent with mock for testing
mock_llm = MockLLMWithChain(chain=[
    "Thought: I need to calculate the price.\nAction: calculate_price\nAction Input: {'quantity': 5, 'unit_price': 10.0}",
    "Thought: The discounted price is 45.0.\nAnswer: The total price with discount is $45.00"
])

agent = SimpleReActAgent(
    llm=mock_llm,
    tools=[tool],
    system_header="You are a pricing assistant."
)

# Run the agent
handler = agent.run(user_msg="What's the price for 5 items at $10 each?")
result = await agent.get_results_from_handler(handler)
print(result["response"])  # "The total price with discount is $45.00"
```

## Production Patterns

### Testing Without API Costs

```python
# Development and testing with mocks
def test_agent_business_logic():
    mock_llm = MockLLMWithChain(chain=[...])
    agent = SimpleReActAgent(llm=mock_llm, tools=[...])
    # Test deterministically without API calls
    
# Production with real LLMs
def create_production_agent():
    from llama_index.llms.openai import OpenAI
    real_llm = OpenAI(model="gpt-4")
    agent = SimpleReActAgent(llm=real_llm, tools=[...])
    return agent
```

### WorkflowHandler Pattern

The toolkit uses LlamaIndex's production WorkflowHandler pattern:

```python
# Production-grade pattern
handler = agent.run(user_msg="Process this request")

# Extract structured results
result = await agent.get_results_from_handler(handler)

# Access all aspects of the execution
print(result["response"])    # Final answer
print(result["reasoning"])   # Step-by-step reasoning
print(result["sources"])     # Tool outputs used
print(result["chat_history"]) # Conversation context
```

### Rule-Based Testing

For more flexible testing scenarios:

```python
from fm_app_toolkit.testing import RuleBasedMockLLM

rules = {
    "price": "Thought: Calculate pricing.\nAction: calculate_price\nAction Input: {...}",
    "inventory": "Thought: Check inventory.\nAction: check_stock\nAction Input: {...}",
    "order": "Thought: Process order.\nAction: place_order\nAction Input: {...}"
}

mock_llm = RuleBasedMockLLM(
    rules=rules,
    default_behavior="direct_answer"
)

# Agent responds intelligently based on query content
response = agent.run(user_msg="What's the price?")  # Triggers price rule
```

## Project Structure

```
fm-app-toolkit/
├── fm_app_toolkit/          # Main package
│   ├── agents/              # Agent implementations
│   │   ├── simple_react.py  # ReAct agent using BaseWorkflowAgent
│   │   ├── sample_tools.py  # Example tools
│   │   └── events.py        # Workflow events
│   ├── testing/             # Testing utilities
│   │   ├── mock_chain.py    # Sequential mock LLM
│   │   ├── mock_echo.py     # Streaming mock LLM
│   │   ├── mock_rule_based.py # Rule-based mock LLM
│   │   └── mocks.py         # Backward compatibility
│   ├── tools.py             # Core tool implementations
│   └── main.py              # FastAPI integration (optional)
├── tests/                   # Comprehensive test suite
│   ├── test_*.py           # 100+ tests demonstrating patterns
│   └── test_utilities.py    # Test helper functions
├── Makefile                 # Development commands
├── pyproject.toml           # Project configuration
└── CLAUDE.md               # Development guide
```

## Testing Patterns

### Deterministic Multi-Step Workflows

```python
def test_complex_business_workflow():
    chain = [
        "Thought: Check inventory first.\nAction: check_inventory\nAction Input: {'item': 'widget'}",
        "Thought: Stock available. Calculate price.\nAction: calculate_price\nAction Input: {'quantity': 10}",
        "Thought: Process the order.\nAction: place_order\nAction Input: {'items': [...]}",
        "Thought: Order placed.\nAnswer: Order #12345 confirmed for $450.00"
    ]
    
    mock_llm = MockLLMWithChain(chain=chain)
    agent = SimpleReActAgent(llm=mock_llm, tools=business_tools)
    
    result = await agent.run("Order 10 widgets")
    assert "Order #12345" in result["response"]
    assert len(result["sources"]) == 3  # Three tools used
```

### Error Handling

```python
def test_handles_api_failures():
    chain = [
        "Thought: Call external API.\nAction: fetch_data\nAction Input: {'id': '123'}",
        "Thought: API failed. Use fallback.\nAction: get_cached_data\nAction Input: {'id': '123'}",
        "Thought: Retrieved from cache.\nAnswer: Here's the cached data..."
    ]
    
    # Test graceful degradation and error recovery
```

## Development Workflow

### Essential Commands

```bash
# Environment Management
make environment-create   # First-time setup
make environment-sync     # Update dependencies

# Development
make format              # Auto-format code
make lint               # Fix linting issues
make type-check         # Type checking

# Testing
make unit-test          # Run all tests
make validate-branch    # Pre-commit validation

# Coverage
make all-test           # Run with coverage report
```

### Adding New Patterns

1. **Create new tools** representing your business logic
2. **Define mock chains** for testing scenarios
3. **Implement agents** using the patterns provided
4. **Test deterministically** with mocks
5. **Deploy with real LLMs** in production

## Why FM App Toolkit?

### For Developers
- **Faster Development** - No waiting for API responses during testing
- **Lower Costs** - Zero API costs during development
- **Better Testing** - Deterministic, reproducible tests
- **Clear Patterns** - Production-ready implementations to build upon

### For Teams
- **CI/CD Ready** - No API keys needed in pipelines
- **Consistent Testing** - Same results across all environments
- **Knowledge Sharing** - Clear patterns for FM app development
- **Production Focus** - Bridge from prototype to production

### For Business
- **Reduced Development Costs** - Minimize API usage during development
- **Faster Time-to-Market** - Rapid iteration and testing
- **Quality Assurance** - Comprehensive testing without external dependencies
- **Scalable Patterns** - Reusable components across projects

## Best Practices

### 1. Test-Driven Development
Write tests with mocks first, then implement with real LLMs:
```python
# 1. Define expected behavior with mocks
# 2. Implement business logic
# 3. Validate with real LLMs
# 4. Deploy to production
```

### 2. Separation of Concerns
Keep business logic in tools, FM interaction in agents:
```python
# Business logic in tools
def calculate_shipping(weight, distance):
    # Pure business logic
    return weight * distance * RATE

# FM orchestration in agents
agent = SimpleReActAgent(tools=[calculate_shipping])
```

### 3. Progressive Enhancement
Start simple, add complexity gradually:
```python
# Start with MockLLMWithChain for basic flows
# Move to RuleBasedMockLLM for dynamic scenarios
# Finally integrate real LLMs for production
```

## Advanced Patterns

### Custom Mock Behaviors

```python
class BusinessMockLLM(MockLLMWithChain):
    """Mock with business-specific behaviors."""
    
    def chat(self, messages, **kwargs):
        # Add business rule validation
        if "urgent" in messages[-1].content.lower():
            # Prioritize urgent requests
            return self.urgent_response()
        return super().chat(messages, **kwargs)
```

### Workflow Composition

```python
# Compose complex workflows from simple agents
pricing_agent = SimpleReActAgent(tools=[pricing_tools])
inventory_agent = SimpleReActAgent(tools=[inventory_tools])
order_agent = SimpleReActAgent(tools=[order_tools])

# Orchestrate multi-agent workflows
async def process_order(request):
    price = await pricing_agent.run(request)
    stock = await inventory_agent.run(request)
    order = await order_agent.run(f"Price: {price}, Stock: {stock}")
    return order
```

## Contributing

We welcome contributions that enhance the toolkit:

- **New Mock Patterns** - Additional testing strategies
- **Agent Implementations** - Different reasoning patterns
- **Tool Libraries** - Common business logic tools
- **Documentation** - Tutorials and guides
- **Integration Examples** - Real-world use cases

## Related Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/) - Official LlamaIndex docs
- [Foundation Models Guide](https://github.com/fm-guide) - FM best practices
- [CLAUDE.md](CLAUDE.md) - Development guidelines for this project

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

---

**FM App Toolkit** - Building production-grade foundation model applications with confidence 🚀

*Concrete patterns. Reliable testing. Production ready.*