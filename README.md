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

### Data Loading with Repository Pattern
Clean abstraction for document loading that enables testing without external services:
- **DocumentRepository** - Abstract interface for consistent document handling
- **LocalDocumentRepository** - Load documents from local filesystem for development/testing
- **GCPDocumentRepository** - Production loading from Google Cloud Storage
- Test with local files, deploy with cloud storage - same interface, zero code changes

*📚 Full article on this pattern coming next week at [AI Enhanced Engineer](https://aienhancedengineer.substack.com/)*

### Mock LLM Framework
Complete mock implementations that extend LlamaIndex's base LLM class:
- **MockLLMWithChain** - Sequential response patterns for multi-step workflows
- **MockLLMEchoStream** - Streaming behavior testing
- **RuleBasedMockLLM** - Dynamic responses based on configurable rules

*See [testing/README.md](fm_app_toolkit/testing/README.md) for detailed documentation*

### Agent Implementations
Production-ready agent patterns:
- **SimpleReActAgent** - Clear implementation of the ReAct pattern using BaseWorkflowAgent
- **WorkflowHandler Pattern** - Production-grade event handling and result extraction
- **Tool Integration** - Seamless connection between agents and business logic

*See [agents/README.md](fm_app_toolkit/agents/README.md) for implementation details*

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

## Basic Usage

### Document Loading

```python
from fm_app_toolkit.data_loading import LocalDocumentRepository, GCPDocumentRepository

# Development: Load from local files
dev_repo = LocalDocumentRepository(input_dir="./data")
documents = dev_repo.load_documents()

# Production: Load from cloud storage  
prod_repo = GCPDocumentRepository(bucket="my-bucket", prefix="docs/")
documents = prod_repo.load_documents()

# Same interface for both - build once, deploy anywhere
```

### Agent with Mock LLM

```python
from fm_app_toolkit.agents import SimpleReActAgent
from fm_app_toolkit.testing import MockLLMWithChain

# Mock LLM for deterministic testing
mock_llm = MockLLMWithChain(chain=[
    "Thought: Calculate the price.\nAction: calculate_price\nAction Input: {'quantity': 5, 'unit_price': 10}",
    "Thought: Done.\nAnswer: Total is $45 with 10% discount"
])

# Create and run agent
agent = SimpleReActAgent(llm=mock_llm, tools=[calculate_price_tool])
result = await agent.run("What's the price for 5 items at $10 each?")
```

## Production Patterns

### Swapping Mocks for Real LLMs

```python
def create_agent(environment="development"):
    if environment == "development":
        # Use mocks for testing
        from fm_app_toolkit.testing import MockLLMWithChain
        llm = MockLLMWithChain(chain=[...])
    else:
        # Use real LLM in production
        from llama_index.llms.openai import OpenAI
        llm = OpenAI(model="gpt-4")
    
    return SimpleReActAgent(llm=llm, tools=[...])
```

### WorkflowHandler Pattern

```python
# LlamaIndex production pattern for structured results
handler = agent.run(user_msg="Process this request")
result = await agent.get_results_from_handler(handler)

# Access execution details
print(result["response"])    # Final answer
print(result["reasoning"])   # Step-by-step reasoning
print(result["sources"])     # Tool outputs
```

### Rule-Based Testing

```python
from fm_app_toolkit.testing import RuleBasedMockLLM

# Dynamic responses based on query content
mock_llm = RuleBasedMockLLM(
    rules={
        "price": "Action: calculate_price",
        "stock": "Action: check_inventory",
    },
    default_behavior="direct_answer"
)
```

## Project Structure

```
fm-app-toolkit/
├── fm_app_toolkit/          # Main package
│   ├── agents/              # Agent implementations
│   ├── data_loading/        # Document loading patterns
│   ├── testing/             # Mock LLM framework
│   └── tools.py            # Core tool implementations
├── tests/                   # 125+ tests demonstrating patterns
├── Makefile                # Development commands
└── CLAUDE.md              # Development guide
```

Each module has its own README with detailed documentation and examples.

## Testing Philosophy

### Write Once, Test Everywhere

```python
def test_business_workflow():
    # Define deterministic test scenario
    mock_llm = MockLLMWithChain(chain=[
        "Thought: Check inventory.\nAction: check_stock",
        "Thought: Calculate price.\nAction: calculate_price",
        "Thought: Place order.\nAnswer: Order #123 confirmed"
    ])
    
    agent = SimpleReActAgent(llm=mock_llm, tools=business_tools)
    result = await agent.run("Order 10 widgets")
    
    assert "Order #123" in result["response"]
    assert len(result["sources"]) == 2  # Two tools used
```

*See [tests/](tests/) for comprehensive examples*

## Development Workflow

### Essential Commands

```bash
# Environment
make environment-create   # First-time setup
make environment-sync     # Update dependencies

# Development
make format              # Auto-format code
make lint               # Fix linting issues
make type-check         # Type checking

# Testing
make unit-test          # Run all tests
make validate-branch    # Pre-commit validation
```

## Best Practices

### 1. Test-Driven Development
Define expected behavior with mocks first, then implement with real LLMs

### 2. Separation of Concerns
Keep business logic in tools, FM orchestration in agents

### 3. Progressive Enhancement
Start with MockLLMWithChain → Add RuleBasedMockLLM → Deploy with real LLMs

## Why FM App Toolkit?

### For Developers
- **Zero API costs during development** - Test with mocks
- **Deterministic testing** - Reproducible results every time
- **Clear patterns** - Production-ready code to build upon

### For Teams
- **CI/CD Ready** - No API keys needed in pipelines
- **Consistent Testing** - Same results across environments
- **Knowledge Sharing** - Clear patterns for FM development

### For Business
- **Reduced Costs** - Minimize API usage during development
- **Faster Iteration** - Rapid testing without external dependencies
- **Quality Assurance** - Comprehensive testing coverage

## Contributing

We welcome contributions:
- **New Mock Patterns** - Additional testing strategies
- **Agent Implementations** - Different reasoning patterns
- **Tool Libraries** - Common business logic tools
- **Documentation** - Tutorials and guides

## Related Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/) - Official LlamaIndex docs
- [AI Enhanced Engineer](https://aienhancedengineer.substack.com/) - Articles on FM patterns
- [CLAUDE.md](CLAUDE.md) - Development guidelines for this project

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

---

**FM App Toolkit** - Building production-grade foundation model applications with confidence 🚀

*Concrete patterns. Reliable testing. Production ready.*