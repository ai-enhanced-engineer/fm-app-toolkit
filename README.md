# FM App Toolkit

**Foundation Model Application Toolkit** - Concrete implementation and testing patterns for developing production-grade foundation model based applications.

## Where This Fits: The Three-Layer AI Stack

In her book ["AI Engineering"](https://www.oreilly.com/library/view/ai-engineering/9781098166298/), Chip Huyen describes the modern AI stack as a pyramid with three interconnected layers. At the foundation lies the infrastructure layer—the massive compute resources, GPUs, and cloud platforms that power everything above. In the middle sits the model layer, where foundation models like GPT, Claude, and Gemini are trained and fine-tuned. At the top, where most of us work, is the application layer.

As noted in [The AI Engineering Stack](https://newsletter.pragmaticengineer.com/p/the-ai-engineering-stack) by Gergely Orosz and Chip Huyen, this application layer "has seen the most action in the last two years, and it's still rapidly evolving." It's where foundation model capabilities meet real-world business needs, where ChatGPT becomes a customer service agent, and where Claude helps developers write better code.

![AI Stack Pyramid - Three Layers: Infrastructure (bottom), Model (middle), Application (top)](assets/images/ai-stack-pyramid.png)
*The AI Stack Pyramid: Each layer depends on the one below, with accessibility increasing as you move up. Source: Adapted from Chip Huyen's AI Engineering framework.*

The pyramid structure reveals an important truth: as you move up the stack, the technology becomes more accessible to non-specialists, but paradoxically, building production-grade applications at this layer presents unique challenges. You're working with models you don't control, infrastructure you don't manage, and outputs that aren't deterministic. This is where the FM App Toolkit comes in.

## The Reality of Building at the Application Layer

Imagine this scenario: You've just discovered Claude's API or GPT-4's capabilities. Within hours, you've prototyped something amazing—an agent that can handle complex customer queries, analyze documents, or automate workflows. It works perfectly in your Jupyter notebook. The future feels bright.

Then you try to move to production.

Suddenly, your tests are flaky because the LLM gives different responses each time. Your development costs spiral as every test run burns through API credits. When your agent makes an unexpected decision, you can't debug why. The elegant prototype code doesn't translate to production—it needs error handling, monitoring, cost controls, and a dozen other things you hadn't considered.

These aren't edge cases or theoretical problems. They're the daily reality of working at the application layer, where you're orchestrating powerful models you don't control. As Andrew Ng and others have noted, the application layer is "the place to be" because of its lower barriers to entry and proximity to end users. But those low barriers don't mean the engineering challenges disappear—they just shift.

This toolkit emerged from real-world experience building foundation model applications in production. It provides concrete, battle-tested solutions to these exact problems, using LlamaIndex as our foundation to ensure compatibility with the broader ecosystem while maintaining clean abstractions that make testing and development practical.

## 🚀 Key Components: Bridging the Layers

### Data Loading with Repository Pattern
**Abstracting Infrastructure Concerns**

At the infrastructure layer, data lives in various places—cloud storage, databases, file systems. The Repository pattern provides a clean abstraction that lets you develop locally and deploy to the cloud without changing your application code. You write once against the interface, then swap implementations based on your environment.

- **DocumentRepository** - Abstract interface for consistent document handling
- **LocalDocumentRepository** - Load documents from local filesystem for development/testing
- **GCPDocumentRepository** - Production loading from Google Cloud Storage

This pattern exemplifies how application-layer code can remain agnostic to infrastructure-layer details, enabling the "build once, deploy anywhere" philosophy that makes rapid iteration possible.

*📚 Full article on this pattern coming next week at [AI Enhanced Engineer](https://aienhancedengineer.substack.com/)*

### Mock LLM Framework
**Simulating the Model Layer for Testing**

When you don't control the model layer, how do you write reliable tests? Our mock LLMs simulate foundation model behavior with perfect determinism, extending LlamaIndex's base LLM class for drop-in compatibility. This approach lets you test your application logic without making costly API calls or dealing with non-deterministic outputs.

- **MockLLMWithChain** - Sequential response patterns for multi-step workflows
- **MockLLMEchoStream** - Streaming behavior testing
- **RuleBasedMockLLM** - Dynamic responses based on configurable rules

These mocks effectively create a controllable "model layer" for development, letting you test edge cases, error conditions, and complex reasoning chains that would be impossible or expensive to test with real models.

*See [testing/README.md](fm_app_toolkit/testing/README.md) for detailed documentation*

### Agent Implementations
**Application-Layer Orchestration**

Agents represent the quintessential application-layer construct—they orchestrate foundation models to accomplish complex tasks. Our implementations demonstrate production-ready patterns that bridge the gap between model capabilities and business requirements.

- **SimpleReActAgent** - Clear, pedagogical implementation of the ReAct pattern using BaseWorkflowAgent
- **WorkflowHandler Pattern** - Production-grade event handling and result extraction
- **Tool Integration** - Seamless connection between agents and business logic

These patterns show how to build reliable, debuggable agents that can reason through problems while maintaining the transparency and control necessary for production systems.

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

## Why This Matters: Democratizing the Application Layer

The beauty of the three-layer model is that each layer becomes progressively more accessible. You don't need millions in capital to work at the application layer—you just need good engineering practices and the right patterns. This toolkit provides both.

### For Developers
Work confidently at the application layer without worrying about the complexities below. Test your logic without burning API credits. Debug your agents' reasoning. Move from prototype to production with patterns that actually work.

### For Teams
Build robust CI/CD pipelines without embedding API keys. Share knowledge through clear, testable patterns. Ensure consistent behavior across development, staging, and production environments.

### For Business
As Chip Huyen notes, the application layer is where business value is created—it's where AI capabilities become real products. This toolkit reduces the cost and risk of building at this layer while maintaining the quality and reliability your users expect.

## Contributing

We welcome contributions:
- **New Mock Patterns** - Additional testing strategies
- **Agent Implementations** - Different reasoning patterns
- **Tool Libraries** - Common business logic tools
- **Documentation** - Tutorials and guides

## Related Resources

### Essential Reading
- [AI Engineering Book](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) - Chip Huyen's comprehensive guide to AI engineering
- [The AI Engineering Stack](https://newsletter.pragmaticengineer.com/p/the-ai-engineering-stack) - Gergely Orosz and Chip Huyen on the modern AI stack
- [Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) - Chip Huyen on platform considerations

### Technical Resources
- [LlamaIndex Documentation](https://docs.llamaindex.ai/) - Official LlamaIndex docs
- [AI Enhanced Engineer](https://aienhancedengineer.substack.com/) - Articles on FM patterns
- [CLAUDE.md](CLAUDE.md) - Development guidelines for this project

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

---

**FM App Toolkit** - Building production-grade foundation model applications with confidence 🚀

*Concrete patterns. Reliable testing. Production ready.*