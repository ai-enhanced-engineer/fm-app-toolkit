"""Rule-based mock LLM that generates responses based on keyword patterns in user input."""

from typing import Any, Dict, Optional, Sequence

from llama_cloud import MessageRole
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.llms.llm import LLM
from pydantic import Field


class RuleBasedMockLLM(LLM):
    """Generates responses based on keyword rules, with intelligent fallback behavior.

    Usage:
        >>> rules = {"weather": "Thought: Check weather.\\nAction: get_weather"}
        >>> mock = RuleBasedMockLLM(rules=rules, default_behavior="use_tool")
        >>> response = mock.chat([ChatMessage(role=MessageRole.USER, content="What's the weather?")])
        >>> assert "get_weather" in response.message.content
    """

    rules: Dict[str, str] = Field(default_factory=dict)
    default_behavior: str = Field(default="direct_answer")
    call_count: int = Field(default=0)

    def __init__(
        self, rules: Optional[Dict[str, str]] = None, default_behavior: str = "direct_answer", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.rules = rules or {}
        self.default_behavior = default_behavior
        self.call_count = 0

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        self.call_count += 1

        # Extract the last user message
        user_msg = ""
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                user_msg = msg.content or ""
                break

        # Check for tool-related context in system message
        has_tools = any("tool" in (msg.content or "").lower() for msg in messages if msg.role == MessageRole.SYSTEM)

        # Apply rules based on content
        response = self._apply_rules(user_msg, has_tools)

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response),
            raw={},
        )

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def _apply_rules(self, user_msg: str, has_tools: bool) -> str:
        user_msg_lower = user_msg.lower()

        # Check each rule for keyword matches
        for keyword, pattern in self.rules.items():
            if keyword.lower() in user_msg_lower:
                # Use format to replace {query} placeholder if present
                return pattern.format(query=user_msg)

        # No rule matched, use default behavior
        if self.default_behavior == "use_tool" and has_tools:
            # Intelligently select a tool based on query keywords
            if "add" in user_msg_lower or "sum" in user_msg_lower or "plus" in user_msg_lower:
                return "Thought: I need to add numbers.\\nAction: add\\nAction Input: {}"
            elif "multiply" in user_msg_lower or "times" in user_msg_lower:
                return "Thought: I need to multiply numbers.\\nAction: multiply\\nAction Input: {}"
            elif "weather" in user_msg_lower:
                return "Thought: I need to check the weather.\\nAction: get_weather\\nAction Input: {}"
            else:
                return "Thought: I'll try to help with this.\\nAnswer: I'll do my best to help."

        elif self.default_behavior == "direct_answer":
            return f"Thought: I can answer directly.\\nAnswer: Response to: {user_msg}"

        elif self.default_behavior == "cannot_answer":
            return "Thought: I cannot answer this with available tools.\\nAnswer: I'm sorry, I cannot help with that."

        else:
            # Custom default behavior
            return f"Thought: Processing request.\\nAnswer: Processed: {user_msg}"

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        def gen() -> ChatResponseGen:
            response = self.chat(messages, **kwargs)
            content = response.message.content or ""
            cumulative = ""
            for char in content:
                cumulative += char
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=cumulative), delta=char)

        return gen()

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        async def gen() -> ChatResponseAsyncGen:
            response = await self.achat(messages, **kwargs)
            content = response.message.content or ""
            cumulative = ""
            for char in content:
                cumulative += char
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=cumulative), delta=char)

        return gen()

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise NotImplementedError("Use chat methods for this mock")

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise NotImplementedError("Use chat methods for this mock")

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise NotImplementedError("Use chat methods for this mock")

    def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise NotImplementedError("Use chat methods for this mock")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=256,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name="mock-llm-rule-based",
        )
