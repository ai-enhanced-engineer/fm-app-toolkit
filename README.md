# AIEE Toolset

**The official AI-Enhanced Engineer toolset** from [aiee.io](https://aiee.io)

Production-grade patterns and battle-tested implementations for building reliable AI applications. Nearly a decade of deployment experience distilled into reusable code.

---

## Why This Exists

Building AI applications is easy. Shipping them to production is hard.

You're working with models you don't control, infrastructure you don't manage, and outputs that aren't deterministic<sup>[1](#ref1)</sup>. Production realities hit fast:

- **Tests become flaky** - Non-deterministic LLM outputs break traditional testing<sup>[3](#ref3)</sup>
- **Development costs explode** - Every test run consumes API credits
- **Debugging is opaque** - Understanding why an agent made a decision feels impossible
- **Prototypes need hardening** - Error handling, monitoring, cost controls, resilience patterns you hadn't considered<sup>[4](#ref4)</sup>

**We've built these systems.** This toolkit distills nearly a decade of production AI deployments into concrete, reusable patterns<sup>[5](#ref5)</sup> using LlamaIndex, PydanticAI, and proven architectural principles.

## 🔧 Key Components

### Data Loading with Repository Pattern
**Abstracting Infrastructure Concerns**

One of the first challenges in building AI applications is managing multiple data sources that feed into the same pipelines or services. Without proper abstraction, this creates tangled code at the very beginning of your project, directly impacting testability and deployment configurability.

The [Repository pattern](https://www.cosmicpython.com/book/chapter_02_repository.html) solves this elegantly. Whether your data lives in cloud storage, databases, or local file systems, you write your application code once against a clean interface. We provide concrete implementations—`DocumentRepository` as the abstract base, `LocalDocumentRepository` for development and testing, and `GCPDocumentRepository` for production cloud deployments. Switch between them with a single configuration change, maintaining the "build once, deploy anywhere" philosophy that makes rapid iteration possible.

### Document Indexing
**Creating Searchable Indexes from Documents**

Once you've loaded your documents, you need to make them searchable. The indexing module provides two fundamental approaches: Vector Store indexes for semantic similarity search and Property Graph indexes for relationship queries.

Our `DocumentIndexer` abstraction allows you to switch between indexing strategies based on your needs. Use `VectorStoreIndexer` when you need to find semantically similar content—perfect for RAG pipelines. Choose `PropertyGraphIndexer` when you need to traverse relationships between entities—ideal for knowledge graphs. Both work seamlessly with our mock framework for deterministic testing.

### Mock LLM Framework
**Simulating the Model Layer for Testing**

We've all heard it: "You can't unit test LLM code." This toolkit proves that wrong. Our mock LLMs provide deterministic responses for unit tests without ever hitting the internet, making your test suite fast, reliable, and free.

The framework extends LlamaIndex's base LLM class for drop-in compatibility. Use `TrajectoryMockLLMLlamaIndex` for sequential multi-step workflows, `MockLLMEchoStream` for testing streaming behavior, or `RuleBasedMockLLM` for dynamic query-based responses. These mocks create a controllable "model layer" for development, enabling you to test edge cases, error conditions, and complex reasoning chains that would be impossible or prohibitively expensive with real models.

### Agent Implementations
**Application-Layer Orchestration**

What is an agent? At its core, an agent is an orchestration layer that receives requests, reasons about next steps, acts by calling tools, observes results, and iterates until complete. This toolkit provides two complementary approaches:

**LlamaIndex ReAct Agents** - Transparent step-by-step reasoning with full observability of the thought process. Perfect when you need to debug decision-making or handle complex multi-step workflows.

**PydanticAI Agents** - Structured output with built-in validation and observability. Ideal when you need guaranteed data formats and type safety.

Both approaches integrate seamlessly with your business logic through tools, handle errors gracefully, and support deterministic testing with our mock framework.

#### Choosing Your Agent Approach

```python
# LlamaIndex ReAct: Step-by-step reasoning
from src.agents.llamaindex import SimpleReActAgent
from src.testing import TrajectoryMockLLMLlamaIndex

mock_llm = TrajectoryMockLLMLlamaIndex(chain=[
    "Thought: I need to calculate this.\nAction: multiply\nAction Input: {'a': 15, 'b': 7}",
    "Thought: Now add 23.\nAction: add\nAction Input: {'a': 105, 'b': 23}",
    "Thought: Done.\nAnswer: 15 × 7 + 23 = 128"
])
agent = SimpleReActAgent(llm=mock_llm, tools=[multiply_tool, add_tool])
result = await agent.run("What is 15 times 7 plus 23?")
# Returns: full reasoning steps + final answer

# PydanticAI: Structured output with validation
from src.agents.pydantic import create_analysis_agent
from pydantic_ai.models.test import TestModel

test_model = TestModel(custom_output_args={
    "sentiment": "positive",
    "confidence": 0.95,
    "key_insights": ["High satisfaction", "Quality product"]
})
agent = create_analysis_agent(model=test_model)
result = await agent.run("This product is amazing!")
# Returns: structured AnalysisResult with validated fields
```

## 🎯 Testing Philosophy

**Deterministic Testing for Non-Deterministic Systems**

Following the principle **"don't mock what you don't own"** from [Architecture Patterns with Python](https://www.cosmicpython.com/book/), we own the abstraction. Our mock LLMs extend framework base classes, creating clean boundaries between business logic and external services.

This approach enables deterministic testing without brittle mocks. Define expected behavior with perfect control, then swap in real LLMs for production—same application code.

```python
def test_business_workflow():
    mock_llm = TrajectoryMockLLMLlamaIndex(chain=[
        "Thought: Check stock.\nAction: check_stock",
        "Thought: Calculate total.\nAction: calculate_price",
        "Thought: Done.\nAnswer: Order #123 confirmed"
    ])

    agent = SimpleReActAgent(llm=mock_llm, tools=business_tools)
    result = await agent.run("Order 10 widgets")

    assert "Order #123" in result["response"]
    assert len(result["sources"]) == 2
```

## 🏭 Production Patterns

### Environment-Based Configuration

Develop with mocks, test with mocks, deploy with real models—same codebase:

```python
def create_agent(environment="development"):
    if environment == "development":
        from src.testing import TrajectoryMockLLMLlamaIndex
        llm = TrajectoryMockLLMLlamaIndex(chain=[...])
    else:
        from llama_index.llms.openai import OpenAI
        llm = OpenAI(model="gpt-4")
    return SimpleReActAgent(llm=llm, tools=[...])

# For structured agents
def create_structured_agent(environment="development"):
    if environment == "development":
        from pydantic_ai.models.test import TestModel
        model = TestModel(custom_output_args={...})
    else:
        model = "openai:gpt-4o"  # Production model string
    return create_analysis_agent(model=model)
```

## 🚀 Quick Start

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

# Examples
make process-documents  # See document loading and chunking in action
```

## Project Structure

```
aiee-toolset/
├── src/          # Main package
│   ├── agents/              # Agent implementations
│   │   ├── llamaindex/     # ReAct pattern with LlamaIndex
│   │   └── pydantic/       # Structured agents with PydanticAI
│   ├── data_loading/        # Document loading patterns
│   ├── indexing/            # Document indexing strategies
│   ├── testing/             # Mock LLM framework
│   └── tools.py            # Core tool implementations
├── tests/                   # 220+ tests demonstrating patterns
├── Makefile                # Development commands
└── CLAUDE.md              # Development guide
```

## 📚 Documentation

Each module has detailed documentation:

- **[Agents](src/agents/README.md)** - LlamaIndex ReAct, LangGraph, PydanticAI implementations
- **[Data Loading](src/data_loading/README.md)** - Repository pattern, GCP/Local implementations
- **[Indexing](src/indexing/README.md)** - Vector store and property graph strategies
- **[Testing](src/mocks/README.md)** - Mock LLM framework for deterministic testing

**Tests as Documentation**: 220+ tests in `tests/` demonstrate real-world usage patterns.

## 🤝 Contributing

This toolkit grows stronger with community input. We especially welcome:
- Battle-tested patterns from your production deployments
- Novel testing strategies for complex agent behaviors
- Industry-specific tool implementations
- Real-world case studies and examples

## Resources

### Essential Reading
- [AI Engineering Book](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) - Chip Huyen's comprehensive guide to AI engineering
- [The AI Engineering Stack](https://newsletter.pragmaticengineer.com/p/the-ai-engineering-stack) - Gergely Orosz and Chip Huyen on the modern AI stack
- [Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) - Chip Huyen on platform considerations
- [Architecture Patterns with Python](https://www.cosmicpython.com/book/) - Harry Percival and Bob Gregory on clean architecture

### Technical Resources
- [LlamaIndex Documentation](https://docs.llamaindex.ai/) - Official LlamaIndex docs
- [PydanticAI Documentation](https://ai.pydantic.dev/) - Official PydanticAI docs
- [CLAUDE.md](CLAUDE.md) - Development guidelines for this project

## References

### Academic Foundations

<a id="ref1"></a><sup>1</sup> SEI/Carnegie Mellon (2024). ["The Challenges of Testing in a Non-Deterministic World"](https://www.sei.cmu.edu/blog/the-challenges-of-testing-in-a-non-deterministic-world/). Analysis showing why non-deterministic systems make bugs "rare, intermittent, and hard to reproduce."

<a id="ref2"></a><sup>2</sup> Google (2024). ["MLOps: Continuous delivery and automation pipelines in machine learning"](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning). Google Cloud Architecture Center. *"The real challenge isn't building an ML model, the challenge is building an integrated ML system and to continuously operate it in production."*

<a id="ref3"></a><sup>3</sup> Faubel, L., Schmid, K. & Eichelberger, H. (2023). ["MLOps Challenges in Industry 4.0"](https://doi.org/10.1007/s42979-023-01934-7). SN Computer Science. Comprehensive analysis of MLOps challenges across different industrial contexts.

<a id="ref4"></a><sup>4</sup> Shankar, S., et al. (2024). ["We Have No Idea How Models will Behave in Production until Production: How Engineers Operationalize Machine Learning"](https://arxiv.org/abs/2403.16795). Study highlighting the experimental nature of ML systems and the challenges of moving from notebooks to production-ready code.

<a id="ref5"></a><sup>5</sup> Sculley, D., et al. (2015). ["Hidden Technical Debt in Machine Learning Systems"](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html). NeurIPS 2015. The seminal paper that introduced the concept of technical debt in ML systems, highlighting how ML systems can incur massive ongoing maintenance costs through boundary erosion, entanglement, and hidden feedback loops.

### Industry Perspectives

- Huyen, Chip (2023). ["Building LLM applications for production"](https://huyenchip.com/2023/04/11/llm-engineering.html). Practical insights on why "it's easy to make something cool with LLMs, but very hard to make something production-ready with them."
- MLOps Community (2024). [MLOps World Conference Proceedings](https://mlops.community/). Latest practices and challenges in deploying ML systems at scale.

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

---

🚀 **Ready to ship production AI?** Start with `make environment-create` and have your first deterministic agent test running in minutes.

*From nearly a decade of production AI deployments. For developers shipping real systems.*
