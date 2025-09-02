# Agent Implementations

Transform your applications with AI agents that can reason, act, and solve complex problems autonomously. This guide shows you how to build production-ready agents using two powerful frameworks, each optimized for different use cases.

## Why Build Agents Instead of Using LLMs Directly?

### The Direct LLM Problem

Imagine asking an LLM: *"What's the current weather in Tokyo and should I pack an umbrella for my trip?"*

**Direct LLM Response:** *"I don't have access to current weather data. As of my last update, Tokyo has a temperate climate..."* 

**Result:** Useless hallucination or honest ignorance.

### The Agent Solution

An agent transforms this interaction:

1. **Receives** your question about Tokyo weather
2. **Reasons** "I need current weather data for Tokyo"  
3. **Acts** by calling a weather API tool
4. **Observes** the result: "22°C, 85% humidity, rain forecast"
5. **Iterates** "Now I can give advice based on real data"

**Agent Response:** *"Current weather in Tokyo is 22°C with rain expected this afternoon. Pack layers and definitely bring an umbrella!"*

### Real-World Agent Applications

**Research Assistant Agent**
- Searches multiple academic databases
- Synthesizes findings from 20+ papers
- Generates literature review with citations
- *Result: Hours of research completed in minutes*

**Customer Service Agent** 
- Checks inventory across warehouses
- Processes return requests automatically  
- Updates customer records in CRM
- *Result: 24/7 support with human-level accuracy*

**Data Analysis Agent**
- Pulls metrics from multiple data sources
- Identifies trends and anomalies
- Generates executive summary with visualizations  
- *Result: Business insights without data team bottleneck*

## What Are Agents?

An agent is an orchestration layer that creates a reasoning loop:

```
User Query → [Reason → Act → Observe] → [Reason → Act → Observe] → Answer
```

This loop continues until the agent has gathered enough information to provide a complete, accurate response.

## 🎯 Learning Path: Choose Your Starting Point

**🚀 New to Agents?** Start with ReAct Pattern → Build weather assistant → Learn debugging

**🏗️ Need Structured Output?** Jump to PydanticAI → Build data extraction agent  

**🔧 Building APIs?** Focus on PydanticAI validation → Integrate with FastAPI

**📊 Want Transparency?** Master LlamaIndex ReAct → Build multi-step research workflows

---

## Module Organization

```
agents/
├── llamaindex/          # Transparent reasoning with ReAct pattern
│   ├── simple_react.py  # Step-by-step decision tracking
│   ├── events.py        # Workflow event coordination  
│   └── sample_tools.py  # Example tools for learning
│
└── pydantic/            # Structured output with validation
    ├── analysis_agent.py   # Text analysis with guaranteed format
    └── extraction_agent.py # Data extraction with type safety
```

---

# Part 1: LlamaIndex ReAct Agents

## The Power of Visible Reasoning

**Core Concept:** See exactly how your agent thinks, step by step.

```
User: "Research renewable energy adoption in Nordic countries"

Agent Reasoning (Visible):
Thought: I need recent data on renewable energy in Nordic countries
Action: search_academic_papers  
Action Input: {"query": "renewable energy adoption Nordic 2023-2024"}
Observation: Found 15 papers with key statistics...

Thought: I should get government policy data too
Action: search_government_reports
Action Input: {"query": "Nordic renewable energy policy 2024"}  
Observation: Retrieved policy documents from Norway, Sweden, Denmark...

Thought: Now I can synthesize a comprehensive response
Answer: Nordic countries lead global renewable adoption with Norway at 85%...
```

## Quick Start: Weather Assistant Agent

Let's build something immediately useful - a weather assistant that gives travel advice.

```python
from fm_app_toolkit.agents.llamaindex import SimpleReActAgent
from fm_app_toolkit.testing import MockLLMWithChain

# For learning: Use deterministic responses
learning_responses = [
    "Thought: I need current weather for the requested city.\nAction: get_weather\nAction Input: {'city': 'Tokyo'}",
    "Thought: Got the weather data. Now I should give clothing recommendations.\nAnswer: It's 22°C and rainy in Tokyo. Pack layers and bring an umbrella!"
]

mock_llm = MockLLMWithChain(chain=learning_responses)

# Create your first agent
agent = SimpleReActAgent(
    llm=mock_llm,
    tools=[get_weather_tool, get_clothing_advice_tool],
    verbose=True  # See the reasoning process
)

# Watch it think and act
handler = agent.run("What should I wear in Tokyo today?")
result = await agent.get_results_from_handler(handler)

print("🤖 Agent Response:", result["response"])
print("🧠 Reasoning Steps:", result["reasoning"])  
print("📊 Sources Used:", result["sources"])
```

**What You'll See:**
- Every reasoning step the agent takes
- Which tools it chooses and why
- The data it gathered to form its answer

## Production Example: Research Agent

```python
from llama_index.llms.openai import OpenAI

# Production setup with real LLM
research_agent = SimpleReActAgent(
    llm=OpenAI(model="gpt-4", temperature=0.1),  # Low temp for consistency
    tools=[
        search_papers_tool,
        search_news_tool, 
        analyze_trends_tool,
        generate_summary_tool
    ],
    max_iterations=10,  # Prevent infinite loops
    verbose=True
)

# Complex multi-step research
result = await research_agent.run(
    "Analyze the latest developments in quantum computing and identify "
    "the top 3 companies likely to achieve commercial breakthrough first"
)
```

**Agent Reasoning Flow:**
1. Searches recent quantum computing papers
2. Finds news about company developments  
3. Analyzes funding and patent trends
4. Cross-references technical capabilities
5. Synthesizes findings into ranked recommendations

## When to Choose LlamaIndex ReAct

✅ **Perfect For:**
- **Debugging Required**: Need to see why agent made specific decisions
- **Complex Workflows**: Multi-step processes with branching logic
- **Audit Trails**: Compliance requires decision documentation  
- **LlamaIndex Ecosystem**: Already using RAG or query engines
- **Learning**: Understanding how agents work internally

❌ **Not Ideal For:**
- Simple single-step tasks
- APIs requiring consistent output format
- Performance-critical applications (reasoning adds overhead)

---

# Part 2: PydanticAI Agents  

## Guaranteed Structure, Every Time

**Core Concept:** Get exactly the data format you need, validated and type-safe.

Instead of hoping your agent returns JSON in the right format, PydanticAI guarantees it:

```python
# What you define
class AnalysisResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)  
    key_insights: List[str] = Field(min_items=1)
    word_count: int

# What you're guaranteed to get
result.output.sentiment     # Always a valid sentiment string
result.output.confidence    # Always a float between 0.0 and 1.0  
result.output.key_insights  # Always a non-empty list of strings
result.output.word_count    # Always an integer
```

## Quick Start: Customer Feedback Analyzer

Perfect for APIs that need reliable data formats:

```python
from fm_app_toolkit.agents.pydantic import create_analysis_agent
from pydantic_ai.models.test import TestModel

# For learning: Control exact outputs
test_model = TestModel(custom_output_args={
    "sentiment": "positive",
    "confidence": 0.92,
    "word_count": 47,
    "key_insights": ["Customer loves product quality", "Delivery exceeded expectations"]
})

agent = create_analysis_agent(model=test_model)

# Analyze customer feedback
result = await agent.run(
    "This product is absolutely amazing! Fast delivery and great quality."
)

# Access validated fields
print(f"Sentiment: {result.output.sentiment}")        # "positive"  
print(f"Confidence: {result.output.confidence}")      # 0.92
print(f"Insights: {result.output.key_insights}")      # List[str]
```

## Production Example: Data Extraction Agent  

```python
from fm_app_toolkit.agents.pydantic import create_extraction_agent

# Extract structured data from unstructured text
agent = create_extraction_agent(model="openai:gpt-4o")

business_text = """
Apple reported Q3 revenue of $394.3 billion, driven by strong iPhone 15 sales. 
CEO Tim Cook noted 15% growth in Services revenue. The company announced 
plans to invest $10 billion in R&D next year.
"""

result = await agent.run(business_text)

# Guaranteed structured output
extracted = result.output
print(f"Companies: {extracted.entities}")     # ["Apple"] 
print(f"Numbers: {extracted.numbers}")       # [394.3, 15, 10]
print(f"Summary: {extracted.summary}")       # Auto-generated summary
```

## Advanced: Multi-Tenant Context Management

PydanticAI excels at dependency injection for complex applications:

```python
@dataclass  
class BusinessContext:
    user_id: str
    subscription_tier: str
    api_rate_limit: int
    custom_instructions: Dict[str, str]

# Context-aware analysis
context = BusinessContext(
    user_id="enterprise_client_001",
    subscription_tier="premium", 
    api_rate_limit=1000,
    custom_instructions={"tone": "formal", "detail_level": "comprehensive"}
)

result = await analysis_agent.run(
    "Analyze this quarterly report",
    deps=context  # Agent adapts behavior based on context
)
```

## When to Choose PydanticAI

✅ **Perfect For:**
- **API Development**: Need consistent response formats
- **Data Extraction**: Converting unstructured → structured data
- **Type Safety**: Compile-time validation requirements
- **Multi-Tenant**: Context-dependent agent behavior
- **Observability**: Built-in Logfire integration

❌ **Not Ideal For:**  
- Learning how agents work (black box reasoning)
- Simple text generation tasks
- When you need to debug agent decision-making

---

# Choosing the Right Framework

## 🤔 Decision Tree

**Start Here: What's Your Primary Goal?**

### "I need to debug why my agent made a decision"
→ **LlamaIndex ReAct** (transparent reasoning)

*Example: Medical diagnosis agent where you need to audit the reasoning chain*

### "I'm building an API that returns structured data"  
→ **PydanticAI** (guaranteed output format)

*Example: Customer support API that categorizes tickets and assigns priority*

### "I want to process documents with citations"
→ **LlamaIndex ReAct** (sources tracking)

*Example: Legal research agent that provides case law citations*

### "I need multi-tenant context management"  
→ **PydanticAI** (dependency injection)

*Example: SaaS platform where agent behavior varies by subscription tier*

## Framework Comparison

| Aspect | LlamaIndex ReAct | PydanticAI |
|--------|------------------|------------|
| **Core Strength** | See agent reasoning | Validate output structure |
| **Output Format** | Text + sources + reasoning trace | Validated Pydantic models |
| **Debugging** | Full step-by-step visibility | Final result only |
| **Tool Integration** | LlamaIndex FunctionTool | Native Python functions |
| **Testing** | MockLLMWithChain | TestModel with custom args |
| **Performance** | Slower (reasoning overhead) | Faster (direct to result) |
| **Type Safety** | Runtime checking | Compile-time validation |
| **Best For** | Research, analysis, debugging | APIs, data extraction, validation |

---

# Testing Strategy: Why Agent Testing Is Different

## The Challenge

Agents are **non-deterministic** by nature - the same input might take different reasoning paths. But your **business logic** should be predictable.

**Without Proper Testing:**
- Slow (API calls for every test)
- Expensive ($$ per test run)  
- Flaky (different responses break tests)
- Unreliable (can't run in CI/CD)

**With Mock Testing:**
- Fast (no network calls)
- Free (no API costs)
- Deterministic (same result every time)
- Reliable (works offline)

## Testing Both Frameworks

### LlamaIndex Testing: Control the Reasoning

```python
from fm_app_toolkit.testing import MockLLMWithChain

def test_research_agent_handles_missing_data():
    # Test error recovery path
    chain = [
        "Thought: I need to search for data.\nAction: search\nAction Input: {'query': 'AI trends'}",
        "Thought: No results found. I should try a broader search.\nAction: search\nAction Input: {'query': 'artificial intelligence'}",
        "Thought: Found some results. I can provide a response now.\nAnswer: Based on available data..."
    ]
    
    mock_llm = MockLLMWithChain(chain=chain)
    agent = SimpleReActAgent(llm=mock_llm, tools=[search_tool])
    
    result = await agent.run("What are the latest AI trends?")
    
    assert "artificial intelligence" in result["reasoning"]  # Used fallback search
    assert len(result["sources"]) > 0  # Found data eventually
```

### PydanticAI Testing: Validate Output Structure  

```python
from pydantic_ai.models.test import TestModel

def test_sentiment_agent_output_format():
    # Guarantee exact output structure
    test_model = TestModel(custom_output_args={
        "sentiment": "negative",
        "confidence": 0.78,
        "key_insights": ["Customer frustrated with shipping", "Product quality concerns"]
    })
    
    agent = create_analysis_agent(model=test_model)
    result = await agent.run("This product was terrible and took weeks to arrive!")
    
    # Type-safe assertions  
    assert isinstance(result.output.sentiment, str)
    assert result.output.sentiment in ["positive", "negative", "neutral"]
    assert 0.0 <= result.output.confidence <= 1.0
    assert len(result.output.key_insights) >= 1
```

## Testing Strategy by Use Case

| Use Case | Testing Approach | What to Test |
|----------|------------------|---------------|
| **API Endpoints** | Mock both frameworks | Output format, error handling |
| **Multi-step Workflows** | Mock LlamaIndex reasoning | Path taken, tools used |
| **Data Extraction** | Mock PydanticAI outputs | Field validation, edge cases |
| **Error Handling** | Mock failure scenarios | Graceful degradation |

---

# Advanced Patterns & Best Practices

## 🚨 Common Pitfalls and Solutions

### The Infinite Loop Problem
```python
# ❌ BAD: Agent gets stuck in reasoning loop
agent = SimpleReActAgent(llm=llm, tools=tools)  # No limits!

# ✅ GOOD: Set boundaries  
agent = SimpleReActAgent(
    llm=llm, 
    tools=tools,
    max_iterations=8,      # Prevent infinite loops
    timeout=30            # Hard timeout
)
```

### The Tool Overuse Problem  
```python
# ❌ BAD: Agent calls tools unnecessarily
"Thought: Let me search for 2+2\nAction: search\nAction Input: {'query': '2+2'}"

# ✅ GOOD: Train with examples that reason first
chain = [
    "Thought: 2+2 is basic math, I don't need tools for this.\nAnswer: 4"
]
```

### The Hallucinated Tool Problem
```python  
# ❌ BAD: Agent tries to use non-existent tools
"Action: magic_tool\nAction Input: {...}"

# ✅ GOOD: Validate tools exist and provide clear descriptions
tools = [
    FunctionTool.from_defaults(
        fn=search_web,
        name="search_web",  # Exact name agent should use
        description="Search the internet for current information. Use for recent events, news, or facts not in training data."
    )
]
```

## Performance Optimization

### When Agents Add Value vs Overhead

**Agents Excel At:**
- Complex decision trees (>3 steps)
- Real-time data integration  
- Context-dependent responses
- Error recovery and retry logic

**Direct LLM Better For:**
- Simple text generation
- Single-step transformations
- Creative writing tasks
- When response time is critical

### Optimization Strategies

```python
# Reduce reasoning overhead
llm = OpenAI(
    model="gpt-4",
    temperature=0.1,      # More deterministic
    max_tokens=500        # Limit reasoning verbosity  
)

# Cache expensive tool calls
@lru_cache(maxsize=1000)
def search_web(query: str) -> str:
    return expensive_api_call(query)

# Use streaming for better perceived performance
agent = SimpleReActAgent(llm=llm, tools=tools, streaming=True)
```

## Quick Start Installation

### Install Dependencies
```bash
# For LlamaIndex agents
uv add llama-index-core

# For PydanticAI agents  
uv add pydantic-ai logfire

# Development and testing
uv add --dev pytest pytest-asyncio
```

### Environment Setup
```python
import os
from pathlib import Path

# Development: Use mocks for fast testing
if os.getenv("ENVIRONMENT") == "development":
    from fm_app_toolkit.testing import MockLLMWithChain
    llm = MockLLMWithChain(chain=your_test_responses)
else:
    # Production: Use real models  
    from llama_index.llms.openai import OpenAI
    llm = OpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
```

---

# Summary: Your Agent Journey

## 🎯 Next Steps Based on Your Goal

### **Learning Agents** (Start Here)
1. Run the weather assistant example with MockLLMWithChain
2. Add a new tool and watch the reasoning change
3. Try breaking it and see how error handling works

### **Building APIs**  
1. Create a PydanticAI agent with your data model
2. Test with TestModel to verify output structure
3. Integrate with FastAPI for production endpoint

### **Research & Analysis**
1. Build a ReAct agent with search tools
2. Add multiple data sources and watch synthesis
3. Use reasoning traces to improve prompts

### **Production Deployment**
1. Start with mocks for development
2. Add error handling and rate limiting  
3. Monitor agent performance with Logfire
4. Scale with async patterns

## Key Takeaways

- **LlamaIndex ReAct**: Choose when you need transparent reasoning, complex workflows, or debugging capability
- **PydanticAI**: Choose when you need guaranteed output structure, type safety, or production APIs
- **Both Frameworks**: Support deterministic testing, work with any LLM provider, integrate with business logic

**The most important skill**: Learning to test agents effectively with mocks. This enables rapid iteration and reliable production deployment.

---

*Ready to build your first agent? Start with the Quick Start examples above and join the conversation in our community!*