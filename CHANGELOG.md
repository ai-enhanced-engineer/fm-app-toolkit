# CHANGELOG

<!-- version list -->

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