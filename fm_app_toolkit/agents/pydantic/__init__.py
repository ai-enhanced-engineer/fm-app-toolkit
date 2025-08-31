"""PydanticAI agents package.

This package contains modular PydanticAI agent implementations demonstrating:
- Structured output validation
- Tool registration and usage
- Dependency injection
- Various agent patterns
"""

# Import all models
# Import agent creation functions
from .analysis_agent import create_analysis_agent
from .extraction_agent import create_extraction_agent

# Import standalone agents (following PydanticAI best practices)

__all__ = [
    # Agent creators (factory pattern)
    "create_extraction_agent",
    "create_analysis_agent",
]