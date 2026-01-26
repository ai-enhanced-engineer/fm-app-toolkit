"""PydanticAI agents for deep research with structured output validation.

This module provides a 4-phase deep research pipeline:
1. Planning - Creates research plans with up to 5 search steps
2. Gathering - Executes web searches in parallel
3. Synthesis - Combines findings into coherent reports
4. Verification - Validates research quality and provides confidence scores
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

# Export exceptions
# Export agent factories (cached and uncached versions)
from .agents import (  # noqa: E402
    clear_agent_cache,
    create_gathering_agent,
    create_plan_agent,
    create_synthesis_agent,
    create_verification_agent,
    get_gathering_agent,
    get_plan_agent,
    get_synthesis_agent,
    get_verification_agent,
)
from .exceptions import (  # noqa: E402
    GatheringError,
    PlanningError,
    ResearchPipelineError,
    SynthesisError,
    VerificationError,
)
from .models import (  # noqa: E402
    ResearchPlan,
    ResearchReport,
    SearchResult,
    SearchStep,
    ValidationResult,
)

# Export orchestration
from .research import run_research  # noqa: E402

__all__ = [
    # Exceptions
    "ResearchPipelineError",
    "PlanningError",
    "GatheringError",
    "SynthesisError",
    "VerificationError",
    # Models
    "SearchStep",
    "ResearchPlan",
    "SearchResult",
    "ResearchReport",
    "ValidationResult",
    # Agent factories (cached - for production)
    "get_plan_agent",
    "get_gathering_agent",
    "get_synthesis_agent",
    "get_verification_agent",
    # Agent factories (uncached - for testing)
    "create_plan_agent",
    "create_gathering_agent",
    "create_synthesis_agent",
    "create_verification_agent",
    "clear_agent_cache",
    # Orchestration
    "run_research",
]
