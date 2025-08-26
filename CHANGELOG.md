# CHANGELOG

<!-- version list -->

## v1.2.0 (2025-08-26)

### Bug Fixes

- Address code review feedback with Pydantic validation
  ([#6](https://github.com/ai-enhanced-engineer/fm-app-toolkit/pull/6),
  [`3731a8d`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/3731a8d4589ae7cd306f38a3c6c5873b27ac670e))

- Correct type annotation for _parse_gcs_uri to dict[str, Any]
  ([#6](https://github.com/ai-enhanced-engineer/fm-app-toolkit/pull/6),
  [`3731a8d`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/3731a8d4589ae7cd306f38a3c6c5873b27ac670e))

- Remove unused import from test_indexing_integration.py
  ([#6](https://github.com/ai-enhanced-engineer/fm-app-toolkit/pull/6),
  [`3731a8d`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/3731a8d4589ae7cd306f38a3c6c5873b27ac670e))

### Documentation

- Enhance README with improved formatting and academic references
  ([`10c78f4`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/10c78f411b60e954e2c984b14a7c7f8ee6efefd5))

- Polish main README and add comprehensive agents documentation
  ([`4e76a24`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/4e76a24d9153712b9a5e92ae54ee3dbf07451163))

- Rewrite README with narrative approach and AI stack context
  ([`215e9ce`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/215e9ce5b9770327c584e6fd10f5453710598387))

### Features

- Add document indexing module with VectorStore and PropertyGraph support
  ([#6](https://github.com/ai-enhanced-engineer/fm-app-toolkit/pull/6),
  [`3731a8d`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/3731a8d4589ae7cd306f38a3c6c5873b27ac670e))

### Refactoring

- Add location parameter to DocumentRepository interface
  ([#6](https://github.com/ai-enhanced-engineer/fm-app-toolkit/pull/6),
  [`3731a8d`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/3731a8d4589ae7cd306f38a3c6c5873b27ac670e))

- Align data_loading and indexing modules for pedagogical consistency
  ([#6](https://github.com/ai-enhanced-engineer/fm-app-toolkit/pull/6),
  [`3731a8d`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/3731a8d4589ae7cd306f38a3c6c5873b27ac670e))

- Improve test structure for pedagogical value
  ([#6](https://github.com/ai-enhanced-engineer/fm-app-toolkit/pull/6),
  [`3731a8d`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/3731a8d4589ae7cd306f38a3c6c5873b27ac670e))

- Relocate test data and add indexing integration tests
  ([#6](https://github.com/ai-enhanced-engineer/fm-app-toolkit/pull/6),
  [`3731a8d`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/3731a8d4589ae7cd306f38a3c6c5873b27ac670e))

- Simplify data loading interface and improve test pedagogy
  ([#6](https://github.com/ai-enhanced-engineer/fm-app-toolkit/pull/6),
  [`3731a8d`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/3731a8d4589ae7cd306f38a3c6c5873b27ac670e))

- Streamline indexing tests for better pedagogical value
  ([#6](https://github.com/ai-enhanced-engineer/fm-app-toolkit/pull/6),
  [`3731a8d`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/3731a8d4589ae7cd306f38a3c6c5873b27ac670e))

### Testing

- Add Pydantic validation tests for indexers
  ([#6](https://github.com/ai-enhanced-engineer/fm-app-toolkit/pull/6),
  [`3731a8d`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/3731a8d4589ae7cd306f38a3c6c5873b27ac670e))


## v1.1.0 (2025-08-10)

### Bug Fixes

- Apply linting and type fixes to data loading module
  ([#5](https://github.com/ai-enhanced-engineer/fm-app-toolkit/pull/5),
  [`84962fa`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/84962fa8561a59d76d32ab7b13e592fd64ab8498))

### Features

- Add Repository pattern for data loading
  ([#5](https://github.com/ai-enhanced-engineer/fm-app-toolkit/pull/5),
  [`84962fa`](https://github.com/ai-enhanced-engineer/fm-app-toolkit/commit/84962fa8561a59d76d32ab7b13e592fd64ab8498))


## v1.0.0 (2025-08-10)

- Initial Release

## v1.0.0 (2025-08-09)

### Major Changes

- Renamed project from ai-base-template to ai-test-lab
- Introduced SimpleReActAgent using LlamaIndex Workflow pattern
- Added comprehensive mock LLM implementations for testing
- Implemented 32+ tests demonstrating LLM testing strategies

### Features

- MockLLMWithChain for deterministic response sequences
- MockLLMEchoStream for testing streaming behavior
- SimpleReActAgent as pedagogical ReAct implementation
- Workflow events for agent communication
- Complete ReActAgent testing patterns

### Documentation

- Updated README with testing strategies and patterns
- Created project-specific CLAUDE.md development guide
- Added Architecture Decision Record (ADR)

### Testing

- 32 comprehensive tests for mock LLMs and agents
- Unit tests for both ReActAgent and SimpleReActAgent
- Streaming and async test patterns

## Previous History

This project was renamed from ai-base-template. For historical context, the project originally started as a general ML/AI template but has been refocused specifically on demonstrating LLM testing strategies.
