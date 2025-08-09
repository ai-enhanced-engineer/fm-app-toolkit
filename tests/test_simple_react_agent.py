"""Tests for the simple ReAct agent implementation.

These tests demonstrate how the minimalistic ReAct agent works with
mock LLMs for deterministic testing.
"""


import pytest

from ai_test_lab.agents.simple_react import SimpleReActAgent, Tool
from ai_test_lab.testing.mocks import MockLLMWithChain
from ai_test_lab.tools import add, divide, multiply, reverse_string, word_count


class TestSimpleReActAgent:
    """Test the simple ReAct agent with various scenarios."""
    
    @pytest.mark.asyncio
    async def test_single_tool_execution(self):
        """Test agent executing a single tool to answer a query."""
        # Create mock LLM with ReAct-formatted responses
        mock_llm = MockLLMWithChain(
            chain=[
                "Thought: I need to add 5 and 3 to get the answer.\nAction: add\nAction Input: {\"a\": 5, \"b\": 3}",
                "Thought: The addition resulted in 8.\nAnswer: The sum of 5 and 3 is 8."
            ]
        )
        
        # Create tool
        add_tool = Tool(
            name="add",
            function=add,
            description="Add two numbers together"
        )
        
        # Create agent
        agent = SimpleReActAgent(
            llm=mock_llm,
            tools=[add_tool],
            verbose=True
        )
        
        # Run agent
        result = await agent.run(user_msg="What is 5 plus 3?")
        
        # Verify result
        assert "8" in result
        assert "sum" in result.lower() or "8" in result
        
    @pytest.mark.asyncio
    async def test_multi_step_reasoning(self):
        """Test agent performing multiple reasoning steps."""
        # Create mock with multi-step reasoning
        mock_llm = MockLLMWithChain(
            chain=[
                "Thought: First, I'll multiply 4 by 5.\nAction: multiply\nAction Input: {\"a\": 4, \"b\": 5}",
                "Thought: That gives us 20. Now I'll add 10.\nAction: add\nAction Input: {\"a\": 20, \"b\": 10}",
                "Thought: The final result is 30.\nAnswer: (4 × 5) + 10 = 30"
            ]
        )
        
        # Create tools
        tools = [
            Tool(name="add", function=add, description="Add two numbers"),
            Tool(name="multiply", function=multiply, description="Multiply two numbers")
        ]
        
        # Create agent
        agent = SimpleReActAgent(
            llm=mock_llm,
            tools=tools,
            verbose=True
        )
        
        # Run agent
        result = await agent.run(user_msg="Calculate (4 * 5) + 10")
        
        # Verify result
        assert "30" in result
        
    @pytest.mark.asyncio
    async def test_direct_answer_without_tools(self):
        """Test agent providing direct answer without using tools."""
        # Create mock that answers directly
        mock_llm = MockLLMWithChain(
            chain=[
                "Thought: This is a greeting, I should respond politely.\nAnswer: Hello! I'm here to help you with calculations and text processing. How can I assist you today?"
            ]
        )
        
        # Create agent with tools (but won't use them)
        agent = SimpleReActAgent(
            llm=mock_llm,
            tools=[Tool(name="add", function=add, description="Add numbers")],
            verbose=True
        )
        
        # Run agent
        result = await agent.run(user_msg="Hello!")
        
        # Verify we got a greeting response
        assert "Hello" in result or "help" in result.lower()
        
    @pytest.mark.asyncio
    async def test_string_manipulation(self):
        """Test agent with string manipulation tools."""
        # Create mock for string operations
        mock_llm = MockLLMWithChain(
            chain=[
                "Thought: I need to reverse the string 'hello'.\nAction: reverse_string\nAction Input: {\"text\": \"hello\"}",
                "Thought: The reversed string is 'olleh'.\nAnswer: The reversed version of 'hello' is 'olleh'."
            ]
        )
        
        # Create string tool
        reverse_tool = Tool(
            name="reverse_string",
            function=reverse_string,
            description="Reverse a string"
        )
        
        # Create agent
        agent = SimpleReActAgent(
            llm=mock_llm,
            tools=[reverse_tool],
            verbose=False
        )
        
        # Run agent
        result = await agent.run(user_msg="Reverse the word 'hello'")
        
        # Verify result
        assert "olleh" in result
        
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test agent handling tool execution errors."""
        # Create mock that will trigger division by zero
        mock_llm = MockLLMWithChain(
            chain=[
                "Thought: I'll divide 10 by 0.\nAction: divide\nAction Input: {\"a\": 10, \"b\": 0}",
                "Thought: There was an error. Division by zero is undefined.\nAnswer: Cannot divide by zero - it's mathematically undefined."
            ]
        )
        
        # Create divide tool
        divide_tool = Tool(
            name="divide",
            function=divide,
            description="Divide one number by another"
        )
        
        # Create agent
        agent = SimpleReActAgent(
            llm=mock_llm,
            tools=[divide_tool],
            verbose=True
        )
        
        # Run agent - should handle error gracefully
        result = await agent.run(user_msg="What is 10 divided by 0?")
        
        # Should mention the error
        assert "zero" in result.lower() or "undefined" in result.lower()
        
    @pytest.mark.asyncio
    async def test_max_iterations_limit(self):
        """Test that agent stops after max iterations."""
        # Create mock that never provides an answer
        mock_llm = MockLLMWithChain(
            chain=[
                "Thought: I need to think about this.",
                "Thought: Still thinking...",
                "Thought: This is complex...",
                "Thought: More thinking needed...",
                "Thought: Almost there...",
                "Thought: Still processing..."
            ]
        )
        
        # Create agent with low max_iterations
        agent = SimpleReActAgent(
            llm=mock_llm,
            tools=[],
            max_iterations=3,
            verbose=False
        )
        
        # Run agent
        result = await agent.run(user_msg="Complex query")
        
        # Should return the max iterations message
        assert "couldn't complete" in result.lower() or "iterations" in result.lower()
        
    @pytest.mark.asyncio
    async def test_multiple_tools_selection(self):
        """Test agent selecting the right tool from multiple options."""
        # Create mock that selects word_count tool
        mock_llm = MockLLMWithChain(
            chain=[
                "Thought: I need to count words in the text.\nAction: word_count\nAction Input: {\"text\": \"hello world test\"}",
                "Thought: There are 3 words.\nAnswer: The text 'hello world test' contains 3 words."
            ]
        )
        
        # Create multiple tools
        tools = [
            Tool(name="add", function=add, description="Add numbers"),
            Tool(name="multiply", function=multiply, description="Multiply numbers"),
            Tool(name="word_count", function=word_count, description="Count words in text"),
            Tool(name="reverse_string", function=reverse_string, description="Reverse a string")
        ]
        
        # Create agent
        agent = SimpleReActAgent(
            llm=mock_llm,
            tools=tools,
            verbose=True
        )
        
        # Run agent
        result = await agent.run(user_msg="How many words in 'hello world test'?")
        
        # Verify result
        assert "3" in result
        

class TestToolFunctionality:
    """Test the Tool dataclass and registry."""
    
    def test_tool_creation(self):
        """Test creating tools."""
        tool = Tool(
            name="calculator",
            function=add,
            description="Adds two numbers"
        )
        
        assert tool.name == "calculator"
        assert tool.function == add
        assert tool.description == "Adds two numbers"
        
        # Test that the function works
        result = tool.function(5, 3)
        assert result == 8
        
    def test_tool_registry(self):
        """Test that tools are properly registered in the agent."""
        tools = [
            Tool(name="add", function=add, description="Add"),
            Tool(name="multiply", function=multiply, description="Multiply")
        ]
        
        mock_llm = MockLLMWithChain(chain=["Answer: Test"])
        agent = SimpleReActAgent(llm=mock_llm, tools=tools)
        
        # Check registry
        assert "add" in agent.tool_registry
        assert "multiply" in agent.tool_registry
        assert agent.tool_registry["add"].function == add
        assert agent.tool_registry["multiply"].function == multiply


class TestWorkflowIntegration:
    """Test the Workflow integration aspects."""
    
    @pytest.mark.asyncio
    async def test_workflow_returns_dict(self):
        """Test that the workflow returns a dictionary with expected keys."""
        mock_llm = MockLLMWithChain(
            chain=["Thought: Simple response.\nAnswer: Test answer"]
        )
        
        agent = SimpleReActAgent(llm=mock_llm, tools=[])
        
        # Run with the underlying workflow method to get full result
        result = await agent.run(user_msg="Test")
        
        # The run method should return the response string for compatibility
        assert isinstance(result, str)
        assert "Test answer" in result
        
    @pytest.mark.asyncio 
    async def test_workflow_with_tools(self):
        """Test workflow execution with tools."""
        mock_llm = MockLLMWithChain(
            chain=[
                "Thought: Adding numbers.\nAction: add\nAction Input: {\"a\": 2, \"b\": 3}",
                "Thought: Got result.\nAnswer: The sum is 5"
            ]
        )
        
        add_tool = Tool(name="add", function=add, description="Add numbers")
        agent = SimpleReActAgent(llm=mock_llm, tools=[add_tool])
        
        result = await agent.run(user_msg="Add 2 and 3")
        assert "5" in result