# FM App Toolkit

**Foundation Model Application Toolkit** - Battle-tested patterns and concrete implementations for building production-grade AI applications.

## Where This Fits: The Three-Layer AI Stack

In her book [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/), Chip Huyen describes the modern AI stack as a pyramid with three interconnected layers. At the foundation lies the infrastructure layer—the massive compute resources, GPUs, and cloud platforms that power everything above. In the middle sits the model layer, where foundation models like GPT, Claude, and Gemini are trained and fine-tuned. At the top, where most of us work, is the application layer—which, as noted in [The AI Engineering Stack](https://newsletter.pragmaticengineer.com/p/the-ai-engineering-stack), has seen explosive growth and is where foundation model capabilities meet real-world business needs.

![AI Stack Pyramid - Three Layers: Infrastructure (bottom), Model (middle), Application (top)](assets/images/ai-stack-pyramid.png)
*The AI Stack Pyramid: Each layer depends on the one below, with accessibility increasing as you move up. Source: Adapted from Chip Huyen's AI Engineering framework.*

The pyramid structure reveals an important truth: as you move up the stack, the technology becomes more accessible to non-specialists, but paradoxically, building production-grade applications at this layer presents unique challenges. You're working with models you don't control, infrastructure you don't manage, and outputs that aren't deterministic. This is where the FM App Toolkit comes in.

## The Reality of Building at the Application Layer

Everyone talks about shipping AI apps to production, but few actually show you how. We've gathered nearly a decade of experience deploying production-grade ML and AI applications, and this repository shares our hard-won insights in a concrete, practical way. Our goal is simple: enable you to rapidly build trustworthy, observable AI applications that can serve real users at scale.

The challenges are real and immediate. Your LLM-powered prototype works perfectly in development, but production is a different beast entirely. Tests become flaky with non-deterministic outputs. Development costs explode as every test run consumes API credits. When your agent makes an unexpected decision, debugging becomes a detective story without clues. The elegant notebook code needs error handling, monitoring, cost controls, and resilience patterns you hadn't considered.

We've been there. We've built these systems. And we've distilled our experience into this toolkit—concrete, battle-tested patterns that bridge the gap between prototype and production. Using LlamaIndex as our foundation ensures compatibility with the broader ecosystem while our abstractions make testing deterministic and development cost-effective.

## 🚀 Key Components: Bridging the Layers

### Data Loading with Repository Pattern
**Abstracting Infrastructure Concerns**

One of the first challenges in building AI applications is managing multiple data sources that feed into the same pipelines or services. Without proper abstraction, this creates tangled code at the very beginning of your project, directly impacting testability and deployment configurability.

The [Repository pattern](https://www.cosmicpython.com/book/chapter_02_repository.html) solves this elegantly. Whether your data lives in cloud storage, databases, or local file systems, you write your application code once against a clean interface. We provide concrete implementations—`DocumentRepository` as the abstract base, `LocalDocumentRepository` for development and testing, and `GCPDocumentRepository` for production cloud deployments. Switch between them with a single configuration change, maintaining the "build once, deploy anywhere" philosophy that makes rapid iteration possible.

*📚 Full article on this pattern coming next week at [AI Enhanced Engineer](https://aienhancedengineer.substack.com/)*

### Mock LLM Framework
**Simulating the Model Layer for Testing**

We've all heard it: "There's simply no way to unit test that—it uses an LLM." This toolkit proves that misconception wrong. Our mock LLMs provide deterministic responses for unit tests without ever hitting the internet, making your test suite fast, reliable, and free.

The framework extends LlamaIndex's base LLM class for drop-in compatibility. Use `MockLLMWithChain` for sequential multi-step workflows, `MockLLMEchoStream` for testing streaming behavior, or `RuleBasedMockLLM` for dynamic query-based responses. These mocks create a controllable "model layer" for development, enabling you to test edge cases, error conditions, and complex reasoning chains that would be impossible or prohibitively expensive with real models.

*See [testing/README.md](fm_app_toolkit/testing/README.md) for detailed documentation*

### Agent Implementations
**Application-Layer Orchestration**

Everyone talks about agents, but what does that actually look like in code? Where do they live in your application architecture? This toolkit answers those questions with concrete, working implementations.

Our `SimpleReActAgent` provides a clear, pedagogical implementation of the ReAct pattern using LlamaIndex's BaseWorkflowAgent—showing exactly how agents reason through problems step by step. The toolkit demonstrates how to integrate tools seamlessly with your business logic, maintain observability throughout the reasoning process, and handle errors gracefully. These aren't theoretical patterns; they're production-tested approaches that give you the transparency and control necessary for real-world systems.

*See [agents/README.md](fm_app_toolkit/agents/README.md) for implementation details*

## Testing Philosophy

**Write Once, Test Everywhere**

Our testing approach is inspired by the principles in [Architecture Patterns with Python](https://www.cosmicpython.com/book/), particularly the rule: "don't mock what you don't own." Instead of mocking external LLM APIs directly, we own the abstraction—our mock LLMs extend LlamaIndex's base class, creating a clean boundary between our code and external services.

This pattern avoids "Mock Hell" where tests become brittle and hard to maintain. By owning the interface, we can test our business logic in isolation with deterministic mocks, while the adapter pattern ensures our core application code remains unchanged even if we switch LLM providers.

The foundation of reliable AI applications is deterministic testing. Our approach lets you define expected agent behavior with perfect control, then swap in real LLMs for production without changing your application code.

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

*See [tests/](tests/) for comprehensive examples with 125+ test cases.*

## Quick Start

### Prerequisites
- Python 3.12+
- Make

### Installation

```bash
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

### Environment-Based Configuration

The key to moving from development to production is clean environment-based configuration. Develop with mocks, test with mocks, deploy with real models—all using the same codebase:

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

## Getting Started with Real Code

The best way to understand these patterns is to see them in action. Explore our [tests/](tests/) directory for 125+ examples of real-world scenarios, or dive into the module-specific documentation:

- [testing/README.md](fm_app_toolkit/testing/README.md) - Deep dive into mock LLM patterns
- [agents/README.md](fm_app_toolkit/agents/README.md) - Agent implementation details
- [data_loading/README.md](fm_app_toolkit/data_loading/README.md) - Repository pattern guide

## Contributing

This toolkit grows stronger with community input. We especially welcome:
- Battle-tested patterns from your production deployments
- Novel testing strategies for complex agent behaviors  
- Industry-specific tool implementations
- Real-world case studies and examples

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

**Ready to build production AI applications?** Start with `make environment-create` and have your first deterministic agent test running in minutes.

*From nearly a decade of production AI deployments. For developers shipping real systems.*