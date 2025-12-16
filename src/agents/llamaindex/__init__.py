"""Simple ReAct agent implementation for educational purposes."""

from dotenv import load_dotenv
from llama_index.core import set_global_handler

from .minimal_react import MinimalReActAgent
from .minimal_react import Tool as MinimalTool
from .simple_react import SimpleReActAgent, Tool

__all__ = ["MinimalReActAgent", "MinimalTool", "SimpleReActAgent", "Tool"]

# Load environment variables for API keys
load_dotenv()

# Set up Langfuse observability for LlamaIndex
set_global_handler("langfuse")
