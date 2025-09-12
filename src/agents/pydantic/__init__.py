"""PydanticAI agents for structured output with validation.

This module provides agents that guarantee structured, validated outputs
using the PydanticAI framework with built-in observability.
"""

from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Configure Logfire observability for all PydanticAI agents
try:
    import logfire

    logfire.configure()
    logfire.instrument_pydantic_ai()
except Exception:
    # Logfire not configured or not available
    pass

# Export main agent factories and types
from .analysis_agent import AnalysisContext, AnalysisResult, create_analysis_agent  # noqa: E402
from .extraction_agent import DataExtraction, create_extraction_agent  # noqa: E402

__all__ = [
    "create_analysis_agent",
    "create_extraction_agent",
    "AnalysisResult",
    "AnalysisContext",
    "DataExtraction",
]
