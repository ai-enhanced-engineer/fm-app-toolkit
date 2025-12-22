# Agent Implementations

This directory contains working code for the **[AI Agents in Production](https://aienhancedengineer.substack.com)** article series. The code is designed for learning—each implementation demonstrates concepts from specific articles.

> **Series Status**: Currently at Article 2.1. Code for later articles (testing, framework comparison) exists but articles are not yet published.

## Module Organization

```
agents/
├── llamaindex/              # LlamaIndex ReAct agents
│   ├── minimal_react.py     # Educational: explicit loop (~230 lines)
│   ├── simple_react.py      # Production: BaseWorkflowAgent (~320 lines)
│   ├── simple_react_examples.py  # Usage examples
│   └── sample_tools.py      # Demo tools (weather, calculate, time)
│
├── langgraph/               # LangGraph ReAct agents
│   ├── minimal_react.py     # Educational: explicit graph (~290 lines)
│   └── simple_react.py      # Production: create_react_agent (~170 lines)
│
└── pydantic/                # PydanticAI structured output
    ├── analysis_agent.py    # Text analysis with guaranteed format
    └── extraction_agent.py  # Data extraction with type safety
```

## Six Canonical Harness Components (Article 2.1)

Each component maps to specific code locations:

| Component | LlamaIndex | LangGraph | PydanticAI |
|-----------|------------|-----------|------------|
| **Reasoning Engine** | `llm` parameter | `llm` parameter | `model` parameter |
| **Planning & Orchestration** | `minimal_react.py` loop | `StateGraph` nodes/edges | Agent `run()` |
| **Tool Registry** | `Tool` dataclass | `@tool` decorator | `@agent.tool` |
| **Memory & Context** | `ctx.store` | `AgentState` reducer | `deps` injection |
| **State & Persistence** | `BaseWorkflowAgent` | `CompiledStateGraph` | Result object |
| **Structured I/O** | ReActOutputParser | Message types | Pydantic models |

## Quick Start

```python
from src.agents.llamaindex import MinimalReActAgent, MinimalTool
from src.testing.mock_chain import TrajectoryMockLLMLlamaIndex

# Deterministic responses for learning
mock_llm = TrajectoryMockLLMLlamaIndex(chain=[
    'Thought: I need weather data.\nAction: get_weather\nAction Input: {"location": "Tokyo"}',
    'Thought: Got it.\nAnswer: Tokyo is 22°C and sunny.'
])

# Simple tool
def get_weather(location: str) -> str:
    return f"{location}: 22°C, sunny"

agent = MinimalReActAgent(
    llm=mock_llm,
    tools=[MinimalTool(name="get_weather", description="Get weather", function=get_weather)],
    verbose=True
)

result = await agent.run("What's the weather in Tokyo?")
# result = {"response": "...", "reasoning": [...], "sources": [...]}
```

## Code Reference by Article

| Article | Topic | Key Files |
|---------|-------|-----------|
| **2** | Agent Loop | `llamaindex/minimal_react.py` (explicit loop) |
| **2.1** | Harness Components | All agent files (see table above) |
| **3** | Testing *(upcoming)* | `src/testing/mock_chain.py`, `src/testing/mock_langchain.py` |
| **4** | Framework Comparison *(upcoming)* | `llamaindex/`, `langgraph/`, `pydantic/` |

## Learning Path

**Understand the loop** → Read `minimal_react.py` (explicit for-loop, regex parsing)

**Use a framework** → Study `simple_react.py` (BaseWorkflowAgent pattern)

**See examples** → Run `simple_react_examples.py`

## Testing

All agents support deterministic testing via mock LLMs:

```python
# LlamaIndex
from src.testing.mock_chain import TrajectoryMockLLMLlamaIndex

# LangGraph
from src.testing.mock_langchain import TrajectoryMockLLMLangChain

# PydanticAI
from pydantic_ai.models.test import TestModel
```

## Further Reading

- **Article Series**: [AI Agents in Production](https://aienhancedengineer.substack.com)
- **Repository**: [fm-app-toolkit](https://github.com/ai-enhanced-engineer/fm-app-toolkit)
- **Design Patterns**: [agentic-design-patterns](https://github.com/ai-enhanced-engineer/agentic-design-patterns)
