"""Custom exceptions for the deep research pipeline.

This module provides a structured exception hierarchy for better error handling
and more informative error messages throughout the research pipeline.
"""


class ResearchPipelineError(Exception):
    """Base exception for research pipeline errors."""


class PlanningError(ResearchPipelineError):
    """Raised when the planning phase fails."""

    def __init__(self, topic: str, reason: str) -> None:
        self.topic = topic
        self.reason = reason
        super().__init__(f"Failed to create research plan for '{topic}': {reason}")


class GatheringError(ResearchPipelineError):
    """Raised when the gathering phase fails completely."""

    def __init__(self, attempted: int, failed: int) -> None:
        self.attempted = attempted
        self.failed = failed
        super().__init__(f"All {attempted} search attempts failed. Cannot proceed with synthesis.")


class SynthesisError(ResearchPipelineError):
    """Raised when the synthesis phase fails."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Failed to synthesize research report: {reason}")


class VerificationError(ResearchPipelineError):
    """Raised when the verification phase fails."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Failed to verify research: {reason}")
