# AI Agent Best Practices (2024-2026)

**Research Date:** 2026-01-25
**Research Mode:** Deep (multi-source synthesis)
**Topics:** ReAct Patterns, Tool Use, Observability, Production Patterns, Multi-Framework

---

## Executive Summary

This document synthesizes current best practices for building AI agents across multiple frameworks (LangGraph, LlamaIndex, PydanticAI) with guidance from Anthropic, Google, and industry practitioners. Key themes emerge:

1. **Start simple** - Single agents with ReAct handle most real-world tasks; add complexity only with clear evidence of benefit
2. **Workflows vs Agents** - Use fixed workflows for predictable tasks, dynamic agents for unpredictable situations
3. **Observability is critical** - Production agents require end-to-end visibility into prompts, tool calls, and decision paths
4. **Fail gracefully** - The primary engineering challenge is not intelligence but predictable failure handling
5. **Context engineering** - Treat context as a first-class system with its own architecture and lifecycle

---

## 1. ReAct Pattern Best Practices

### 1.1 Core Pattern Overview

The ReAct (Reasoning + Acting) pattern introduced by [Yao et al., 2022](https://www.promptingguide.ai/techniques/react) structures agent behavior into explicit reasoning loops:

```
Thought -> Action -> Observation -> (repeat until done)
```

**Key Benefits:**
- Every decision becomes visible, creating a clear audit trail
- When agents fail, you see exactly where logic breaks down
- Provides enough structure through reasoning while maintaining flexibility

### 1.2 When to Use ReAct

| Use Case | Fit |
|----------|-----|
| Complex tasks requiring continuous planning | Excellent |
| Research agents following evidence threads | Excellent |
| Debugging assistants with hypothesis testing | Excellent |
| Simple, predictable workflows | Consider alternatives |
| Speed-critical applications | Consider alternatives |

Source: [MetaDesign Solutions - ReAct Best Practices](https://metadesignsolutions.com/using-the-react-pattern-in-ai-agents-best-practices-pitfalls-implementation-tips/)

### 1.3 Known Limitations

- **Latency**: Each reasoning loop requires an additional model call
- **Cost**: Increased token usage from repeated prompts and context
- **Error propagation**: Incorrect tool data can cascade through subsequent reasoning
- **Model dependency**: Effectiveness depends on underlying model's reasoning capability

Source: [Machine Learning Mastery - Agentic AI Design Patterns](https://machinelearningmastery.com/7-must-know-agentic-ai-design-patterns/)

### 1.4 Alternative: ReWOO Pattern

ReWOO (Reasoning Without Observation) generates the entire plan upfront with placeholders, then executes all steps at once. Consider when:
- Speed is critical
- Tasks are well-structured with predictable dependencies
- Token cost is a primary concern

---

## 2. Tool Use Patterns

### 2.1 Tool Definition Best Practices (Anthropic)

From [Anthropic's Tool Use Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use):

**Essential Tool Definition Fields:**
```json
{
  "name": "get_stock_price",
  "description": "Retrieves the current stock price for a given ticker symbol...",
  "input_schema": {
    "type": "object",
    "properties": {
      "ticker": {
        "type": "string",
        "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
      }
    },
    "required": ["ticker"]
  }
}
```

**Critical Guidelines:**
1. **Provide extremely detailed descriptions** - This is the most important factor. Aim for 3-4+ sentences per tool
2. **Explain what, when, and how** - Cover what the tool does, when to use it, parameter meanings, and limitations
3. **Use `input_examples` for complex tools** - Show concrete patterns for well-formed calls
4. **Validate inputs** - Use JSON Schema and Pydantic models for type safety

### 2.2 Tool Error Handling

From [GoCodeo - Error Recovery Strategies](https://www.gocodeo.com/post/error-recovery-and-fallback-strategies-in-ai-agent-development):

```python
# Tool result with error handling
{
  "type": "tool_result",
  "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
  "content": "ConnectionError: weather service unavailable (HTTP 500)",
  "is_error": true
}
```

**Fault Tolerance Requirements:**
- API schema validation
- Automatic retries with exponential backoff
- Circuit breakers to prevent cascading failures
- Well-defined fallback logic

### 2.3 Tool Use Examples (Beta Feature)

Anthropic recommends providing `input_examples` for complex tools:

```python
"input_examples": [
    {"location": "San Francisco, CA", "unit": "fahrenheit"},
    {"location": "Tokyo, Japan", "unit": "celsius"},
    {"location": "New York, NY"}  # Demonstrates optional parameter
]
```

### 2.4 PydanticAI Tool Patterns

From [PydanticAI Documentation](https://ai.pydantic.dev/tools/):

```python
from pydantic_ai import Agent, tool

@agent.tool
def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather in a given location.

    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: Temperature unit, either 'celsius' or 'fahrenheit'
    """
    return json.dumps({"temperature": "20C", "condition": "Sunny"})
```

**Best Practices:**
- Write clear, descriptive docstrings (serve as LLM documentation)
- Use type hints and Pydantic models for validation
- Test tools independently before integration

---

## 3. Agent Observability

### 3.1 Why Observability Matters

From [Maxim AI Observability Guide](https://www.getmaxim.ai/articles/the-best-ai-observability-tools-in-2025-maxim-ai-langsmith-arize-helicone-and-comet-opik/):

> "To build for scale, developers need clear visibility into what's happening within the details of their agent loops. You need to see inputs, trajectories, and outputs, otherwise you won't know what users are asking, how the agent is handling it, and if users are happy with the outcome."

### 3.2 Platform Comparison

| Platform | Best For | Key Features |
|----------|----------|--------------|
| **LangSmith** | LangChain/LangGraph users | Auto-instrumentation, visual trace view, token/latency metrics |
| **Arize Phoenix** | RAG-heavy apps, local-first | Open-source, OpenTelemetry native, embedding visualizations |
| **Arize AX** | Enterprise scale | ML observability, drift detection, data lake integration |

Source: [ZenML - LLM Monitoring Tools](https://www.zenml.io/blog/best-llm-monitoring-tools)

### 3.3 Key Observability Capabilities

**Essential Tracing:**
- Prompts and templates used
- Retrieved context
- Tool selection logic and parameters
- Results returned and exceptions
- Token consumption, latency, and cost per step

**LangSmith Integration:**
```python
# Set one environment variable and tracing works automatically
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
```

### 3.4 Standards and Interoperability

OpenTelemetry is emerging as the standard for AI agent telemetry:
- Phoenix and OpenLLMetry export traces in standard formats
- Multiple platforms can consume the same telemetry data
- Enables mixing specialized tools (e.g., Phoenix for evaluation + Portkey for routing)

---

## 4. Production Agent Patterns

### 4.1 Orchestrator Patterns

From [LangGraph Production Guide](https://www.blog.langchain.com/is-langgraph-used-in-production/):

**Orchestrator-Worker Pattern:**
- Central orchestrator handles global planning, delegation, and state
- Keep orchestrator tool permissions narrow (mostly read and route)
- Workers are specialized for specific tasks

**Hierarchical Multi-Agent:**
```
Supervisor Agent
    |
    +-- Research Agent
    +-- Analysis Agent
    +-- Writing Agent
```

Companies using this pattern: LinkedIn, Uber, Elastic, AppFolio

### 4.2 State Management

From [Kubiya - Context Engineering](https://www.kubiya.ai/blog/context-engineering-ai-agents):

**Hierarchical Memory Architecture:**
- **Short-term**: Recent conversation turns verbatim
- **Medium-term**: Compressed summaries of recent sessions
- **Long-term**: Key facts and relationships from history

**Four Core Strategies:**
1. **Write**: Save context externally (scratchpads, memories)
2. **Select**: Pull relevant context into window
3. **Compress**: Reduce size while maintaining information
4. **Isolate**: Separate context across agents

### 4.3 Retry and Recovery Patterns

From [Sparkco - Retry Logic Best Practices](https://sparkco.ai/blog/mastering-retry-logic-agents-a-deep-dive-into-2025-best-practices):

**Adaptive Retry Strategy:**
```python
# Exponential backoff with jitter
import random
import time

def retry_with_backoff(func, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except TransientError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
```

**Critical Distinctions:**
- **Transient errors**: Retry with backoff (network timeouts, rate limits)
- **Permanent errors**: Fail immediately (invalid input, auth failures)

**Alternative Perspective (UiPath):**
> "Avoid retry mechanisms for agents because agent output isn't deterministic. Instead, capture and handle errors within the agent or tool itself."

### 4.4 Guardrails

From [Vellum - Agentic Workflows Guide](https://www.vellum.ai/blog/agentic-workflows-emerging-architectures-and-design-patterns):

**Essential Guardrails:**
1. **Iteration limits**: Prevent infinite loops
2. **Human bottleneck planning**: Define approval workflows
3. **Success metrics**: Define before adding complexity
4. **Resource caps**: Limit tokens, API calls, execution time

**Implementation:**
```python
# Schema-driven validation
from pydantic import BaseModel, validator

class AgentOutput(BaseModel):
    answer: str
    confidence: float

    @validator('confidence')
    def confidence_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be 0-1')
        return v
```

### 4.5 Human-in-the-Loop

From [LangGraph Human-in-the-Loop](https://docs.langchain.com/oss/python/langgraph/workflows-agents):

**Essential UX Patterns:**
- Approve or reject actions before execution
- Edit the next action
- Ask clarifying questions
- Time travel to re-do from previous steps

**PydanticAI Implementation:**
```python
from pydantic_ai import Agent

agent = Agent(
    model="claude-sonnet-4-5",
    tool_approval_mode="requires_approval"  # Flag tools for human review
)
```

---

## 5. Multi-Framework Considerations

### 5.1 Framework Selection Guide

| Framework | Strengths | Best For |
|-----------|-----------|----------|
| **LangGraph** | Stateful abstractions, time-travel debugging, LangSmith integration | Production multi-agent, complex state |
| **LlamaIndex** | RAG excellence, data connectors, document pipelines | Data-intensive workflows, RAG systems |
| **PydanticAI** | Type safety, FastAPI-like DX, full mypy support | Type-safe agents, production Python |
| **CrewAI** | Role-based agents, simple multi-agent | Multi-role workflows |
| **Anthropic SDK** | Direct Claude access, MCP integration | Claude-native agents |

### 5.2 LlamaIndex Multi-Agent Patterns

From [LlamaIndex Multi-Agent Documentation](https://developers.llamaindex.ai/python/framework/understanding/agent/multi_agent/):

**Pattern 1: AgentWorkflow (Swarm)**
```python
from llama_index.core.agent.workflow import AgentWorkflow

workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent
)
```

**Pattern 2: Orchestrator Agent**
- Expose each agent's `run` method as a tool
- Single orchestrator makes all decisions
- Full control over call sequence

**Pattern 3: Custom Planner**
- LLM outputs structured plan (XML/JSON)
- Python parses and executes imperatively
- Maximum flexibility for external integrations

### 5.3 LangGraph vs LlamaIndex

From [Pedro Azevedo Comparison](https://medium.com/@pedroazevedo6/langgraph-vs-llamaindex-workflows-for-building-agents-the-final-no-bs-guide-2025-11445ef6fadc):

| Aspect | LangGraph | LlamaIndex Workflows |
|--------|-----------|---------------------|
| **Mental Model** | Graph/state machine | Event-driven, Pythonic |
| **Learning Curve** | Steeper | Gentler |
| **RAG Support** | Good | Excellent (specialized) |
| **Observability** | LangSmith (deep) | Standard Python |
| **Scale-off Risk** | Low | Low |

### 5.4 Interoperability

**Current State:**
- LlamaIndex tools can integrate into CrewAI multi-agent setups
- LlamaIndex for retrieval + LangGraph for orchestration works
- CrewAI agents can call LangChain tools

**Emerging Standards:**
- OpenAI function calling format becoming standard
- MCP (Model Context Protocol) for tool standardization
- Agent2Agent (A2A) for inter-agent communication

---

## 6. Claude Agent SDK Patterns

### 6.1 MCP (Model Context Protocol) Integration

From [Anthropic Agent SDK Documentation](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk):

```python
from claude_code_sdk import Agent

agent = Agent(
    mcp_servers=[
        {"type": "stdio", "command": "my-mcp-server"},
        {"type": "http", "url": "https://api.example.com/mcp"}
    ]
)
```

**Transport Types:**
- Local processes (stdio)
- HTTP/SSE connections
- In-process execution

### 6.2 Custom Tools as In-Process MCP

```python
from claude_code_sdk import custom_tool

@custom_tool
def search_documents(query: str) -> str:
    """Search internal documents for information."""
    # Custom implementation
    return results
```

### 6.3 Agent Skills Pattern

From [Anthropic Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills):

**Progressive Disclosure Design:**
- Metadata pre-loaded at startup
- Files read on-demand
- Scripts executed efficiently (only output consumes tokens)

**Benefits:**
- Efficient context usage
- Scalable to thousands of tools
- Dynamic capability loading

### 6.4 Self-Evaluation Approaches

> "Agents that can check and improve their own output are fundamentally more reliable - they catch mistakes before they compound, self-correct when they drift, and get better as they iterate."

**Implementation:**
- Code linting as rules-based feedback
- Visual feedback via Playwright screenshots
- LLM-as-judge for quality assessment

---

## 7. Agentic Loop Design Patterns

### 7.1 Standard Loop Structure

```
1. Planning - Derive current sub-task from query
2. Tool Invocation - Select and execute tools
3. Observation - Collect and interpret outputs
4. State Update - Integrate into memory/context
5. Completion Check - Output result or continue
```

### 7.2 Pattern Selection Guide

| Pattern | When to Use |
|---------|-------------|
| **ReAct** | Complex tasks requiring continuous adaptation |
| **Planning Agent** | Tasks decomposable into structured roadmaps |
| **Reflection** | Quality-critical outputs needing self-evaluation |
| **Controlled Flow** | Well-defined tasks with predictable steps |
| **ReWOO** | Speed-critical with predictable dependencies |

### 7.3 Controlled Flows

From [MongoDB Agentic Patterns](https://medium.com/mongodb/here-are-7-design-patterns-for-agentic-systems-you-need-to-know-d74a4b5835a5):

> "Controlled flows offer a low-risk method to integrate LLMs into workflows. LLMs perform tasks like content generation and analysis within each step, but the sequence of steps and rules for moving between them are fixed by design."

---

## 8. Production Readiness Checklist

### 8.1 Pre-Production Requirements

- [ ] Observability system deployed and tested
- [ ] Error handling framework with graceful degradation
- [ ] Security controls and access policies
- [ ] Cost monitoring with automated alerts and hard limits
- [ ] Rollback mechanisms tested

### 8.2 Operational Expectations

From [Medium - Why AI Agents Fail](https://medium.com/@michael.hannecke/why-ai-agents-fail-in-production-what-ive-learned-the-hard-way-05f5df98cbe5):

> "Production teams need to genuinely tolerate 3-15% error rates and non-deterministic behavior, and have budget for both implementation and ongoing costs."

### 8.3 Key Metrics to Track

- **Task completion rate**: Target > 85% for production
- **Error rate by type**: Transient vs permanent failures
- **Latency percentiles**: p50, p95, p99
- **Token consumption**: Per task type
- **Human escalation rate**: Should decrease over time
- **Cost per task**: Include retries and failures

---

## Sources

### Official Documentation
- [Anthropic Tool Use Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use)
- [Anthropic Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- [Anthropic Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [LangGraph Workflows and Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [LlamaIndex Multi-Agent Patterns](https://developers.llamaindex.ai/python/framework/understanding/agent/multi_agent/)
- [PydanticAI Agents](https://ai.pydantic.dev/agents/)
- [PydanticAI Tools](https://ai.pydantic.dev/tools/)

### Industry Resources
- [Google Cloud - Agentic AI Design Patterns](https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system)
- [Machine Learning Mastery - Agentic AI Design Patterns](https://machinelearningmastery.com/7-must-know-agentic-ai-design-patterns/)
- [ReAct Prompting Guide](https://www.promptingguide.ai/techniques/react)
- [Maxim AI - Observability Tools 2025](https://www.getmaxim.ai/articles/the-best-ai-observability-tools-in-2025-maxim-ai-langsmith-arize-helicone-and-comet-opik/)
- [ZenML - LLM Monitoring Tools](https://www.zenml.io/blog/best-llm-monitoring-tools)

### Production Experience
- [LangChain Blog - Top 5 LangGraph Agents in Production 2024](https://www.blog.langchain.com/top-5-langgraph-agents-in-production-2024/)
- [LangChain Blog - Is LangGraph Used in Production?](https://www.blog.langchain.com/is-langgraph-used-in-production/)
- [Galileo AI - Why Most AI Agents Fail](https://galileo.ai/blog/why-most-ai-agents-fail-and-how-to-fix-them)
- [Medium - Why AI Agents Fail in Production](https://medium.com/@michael.hannecke/why-ai-agents-fail-in-production-what-ive-learned-the-hard-way-05f5df98cbe5)
- [UiPath - 10 Best Practices for Reliable AI Agents](https://www.uipath.com/blog/ai/agent-builder-best-practices)

### Error Handling and Resilience
- [Sparkco - Retry Logic Best Practices 2025](https://sparkco.ai/blog/mastering-retry-logic-agents-a-deep-dive-into-2025-best-practices)
- [GoCodeo - Error Recovery Strategies](https://www.gocodeo.com/post/error-recovery-and-fallback-strategies-in-ai-agent-development)
- [Salesforce - Why AI Agents Fail in Production](https://www.salesforce.com/blog/ai-agent-rag/)

### Context Engineering
- [Maxim AI - Context Window Management](https://www.getmaxim.ai/articles/context-window-management-strategies-for-long-context-ai-agents-and-chatbots/)
- [Kubiya - Context Engineering for AI Agents](https://www.kubiya.ai/blog/context-engineering-ai-agents)
- [Weaviate - Context Engineering](https://weaviate.io/blog/context-engineering)

### Framework Comparisons
- [Pedro Azevedo - LangGraph vs LlamaIndex 2025](https://medium.com/@pedroazevedo6/langgraph-vs-llamaindex-workflows-for-building-agents-the-final-no-bs-guide-2025-11445ef6fadc)
- [Turing - AI Agent Frameworks Comparison 2025](https://www.turing.com/resources/ai-agent-frameworks)
- [Xenoss - LangChain vs LangGraph vs LlamaIndex](https://xenoss.io/blog/langchain-langgraph-llamaindex-llm-frameworks)

---

## Limitations

This research synthesis has the following limitations:

1. **Rapidly evolving field**: Best practices may change within months as frameworks iterate
2. **Framework version specificity**: Some guidance may be version-specific (e.g., LlamaIndex Workflows 1.0, LangGraph 2024-2025)
3. **Use case dependency**: Optimal patterns vary significantly by domain and requirements
4. **Limited empirical data**: Production failure rates and metrics are self-reported
5. **Anthropic focus**: Claude-specific guidance may not generalize to other LLM providers

**Recommended refresh interval:** 3-6 months for production implementations.
