"""Simple ReAct agent implementation for educational purposes."""

from dotenv import load_dotenv
from llama_index.core import set_global_handler

from src.agents.common import Tool

from .minimal_react import MinimalReActAgent
from .simple_react import SimpleReActAgent

__all__ = ["MinimalReActAgent", "SimpleReActAgent", "Tool"]

# Load environment variables for API keys
load_dotenv()

# Set up Langfuse observability for LlamaIndex
set_global_handler("langfuse")
