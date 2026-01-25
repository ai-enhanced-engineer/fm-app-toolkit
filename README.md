# AIEE Toolset

**The official AI-Enhanced Engineer toolset** from [aiee.io](https://aiee.io)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-220%2B%20passing-green)](tests/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ai-enhanced-engineer/aiee-toolset/ci.yml?branch=main)](https://github.com/ai-enhanced-engineer/aiee-toolset/actions)

Production-grade patterns and battle-tested implementations for building reliable AI applications. Nearly a decade of deployment experience distilled into reusable code.

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [🚀 Quick Start](#-quick-start)
- [🔧 Key Features](#-key-features)
- [🎯 Testing Philosophy](#-testing-philosophy)
- [🏭 Production Patterns](#-production-patterns)
- [📂 Project Structure](#-project-structure)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)
- [📖 Resources](#-resources)
- [📝 References](#-references)
- [📜 License](#-license)

---

## Why This Exists

Building AI applications is easy. Shipping them to production is hard.

You're working with models you don't control, infrastructure you don't manage, and outputs that aren't deterministic<sup>[1](#ref1)</sup>. Production realities hit fast:

- **Tests become flaky** - Non-deterministic LLM outputs break traditional testing<sup>[3](#ref3)</sup>
- **Development costs explode** - Every test run consumes API credits
- **Debugging is opaque** - Understanding why an agent made a decision feels impossible
- **Prototypes need hardening** - Error handling, monitoring, cost controls, resilience patterns you hadn't considered<sup>[4](#ref4)</sup>

**We've built these systems.** This toolkit distills nearly a decade of production AI deployments into concrete, reusable patterns<sup>[5](#ref5)</sup> using LlamaIndex, PydanticAI, and proven architectural principles.

## 🚀 Quick Start

### Essential Commands

```bash
# Environment
just init               # First-time setup (installs uv, Python, dependencies, pre-commit)
just sync               # Update dependencies

# Development
just format             # Auto-format code
just lint               # Fix linting issues
just type-check         # Type checking

# Testing
just test               # Run tests with coverage (excludes integration)
just test-integration   # Run integration tests
just test-all           # Run all tests
just validate-branch    # Full validation: format, lint, type-check, test

# Examples
just process-documents  # See document loading and chunking in action
just pydantic-analysis  # Run PydanticAI analysis agent
just llamaindex-react   # Run LlamaIndex ReAct agent
```

## 🔧 Key Features

### Data Loading with Repository Pattern

Abstract your data sources using the [Repository pattern](https://www.cosmicpython.com/book/chapter_02_repository.html). Write once against `DocumentRepository`, deploy anywhere with `LocalDocumentRepository` (dev/test) or `GCPDocumentRepository` (production). → See [architecture diagram](src/data_loading/README.md#architecture)

### Document Indexing

Make documents searchable with `VectorStoreIndexer` (semantic similarity for RAG) or `PropertyGraphIndexer` (relationship traversal for knowledge graphs). Switch strategies with a clean `DocumentIndexer` interface.

### Mock LLM Framework

Test AI applications deterministically without API costs. Use `TrajectoryMockLLMLlamaIndex` for multi-step workflows, `MockLLMEchoStream` for streaming, or `RuleBasedMockLLM` for dynamic responses—all extending framework base classes for drop-in compatibility.

### Agent Implementations

Choose **LlamaIndex ReAct** for transparent step-by-step reasoning with full observability, or **PydanticAI** for structured output with type-safe validation. Both integrate with your tools and support deterministic testing. → See [comparison diagram](src/agents/README.md#architecture)

## 🎯 Testing Philosophy

Following **"don't mock what you don't own"** from [Architecture Patterns with Python](https://www.cosmicpython.com/book/), our mock LLMs extend framework base classes, creating clean boundaries between business logic and external services. Define expected behavior with perfect control for tests, then swap in real LLMs for production—same application code. → See [flow diagram](src/mocks/README.md#flow)

## 🏭 Production Patterns

Develop with mocks, test with mocks, deploy with real models—same codebase. Switch between environments with simple configuration, keeping business logic unchanged. See module documentation for implementation examples.

## 📂 Project Structure

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
├── justfile                # Development commands (run `just` to see all)
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

## 📖 Resources

### Essential Reading
- [AI Engineering Book](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) - Chip Huyen's comprehensive guide to AI engineering
- [The AI Engineering Stack](https://newsletter.pragmaticengineer.com/p/the-ai-engineering-stack) - Gergely Orosz and Chip Huyen on the modern AI stack
- [Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) - Chip Huyen on platform considerations
- [Architecture Patterns with Python](https://www.cosmicpython.com/book/) - Harry Percival and Bob Gregory on clean architecture

### Technical Resources
- [LlamaIndex Documentation](https://docs.llamaindex.ai/) - Official LlamaIndex docs
- [PydanticAI Documentation](https://ai.pydantic.dev/) - Official PydanticAI docs
- [CLAUDE.md](CLAUDE.md) - Development guidelines for this project

## 📝 References

### Academic Foundations

<a id="ref1"></a><sup>1</sup> SEI/Carnegie Mellon (2024). ["The Challenges of Testing in a Non-Deterministic World"](https://www.sei.cmu.edu/blog/the-challenges-of-testing-in-a-non-deterministic-world/). Analysis showing why non-deterministic systems make bugs "rare, intermittent, and hard to reproduce."

<a id="ref2"></a><sup>2</sup> Google (2024). ["MLOps: Continuous delivery and automation pipelines in machine learning"](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning). Google Cloud Architecture Center. *"The real challenge isn't building an ML model, the challenge is building an integrated ML system and to continuously operate it in production."*

<a id="ref3"></a><sup>3</sup> Faubel, L., Schmid, K. & Eichelberger, H. (2023). ["MLOps Challenges in Industry 4.0"](https://doi.org/10.1007/s42979-023-01934-7). SN Computer Science. Comprehensive analysis of MLOps challenges across different industrial contexts.

<a id="ref4"></a><sup>4</sup> Shankar, S., et al. (2024). ["We Have No Idea How Models will Behave in Production until Production: How Engineers Operationalize Machine Learning"](https://arxiv.org/abs/2403.16795). Study highlighting the experimental nature of ML systems and the challenges of moving from notebooks to production-ready code.

<a id="ref5"></a><sup>5</sup> Sculley, D., et al. (2015). ["Hidden Technical Debt in Machine Learning Systems"](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html). NeurIPS 2015. The seminal paper that introduced the concept of technical debt in ML systems, highlighting how ML systems can incur massive ongoing maintenance costs through boundary erosion, entanglement, and hidden feedback loops.

### Industry Perspectives

- Huyen, Chip (2023). ["Building LLM applications for production"](https://huyenchip.com/2023/04/11/llm-engineering.html). Practical insights on why "it's easy to make something cool with LLMs, but very hard to make something production-ready with them."
- MLOps Community (2024). [MLOps World Conference Proceedings](https://mlops.community/). Latest practices and challenges in deploying ML systems at scale.

## 📜 License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

---

🚀 **Ready to ship production AI?** Start with `just init` and have your first deterministic agent test running in minutes.

*From nearly a decade of production AI deployments. For developers shipping real systems.*
