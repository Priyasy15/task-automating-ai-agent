"""Provider-agnostic LLM client abstraction and Google Gemini implementation."""

from abc import ABC, abstractmethod
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from agent.logger import AgentLogger
from agent.tools.base import Tool

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
except ImportError:
    genai = None
    types = None
    APIError = Exception


class ToolCall(BaseModel):
    """Represents a tool call requested by the model."""

    id: str = Field(default="")
    name: str = Field(..., description="The name of the tool to execute.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="The arguments passed to the tool.",
    )


class ChatMessage(BaseModel):
    """Provider-agnostic message in conversation history."""

    model_config = {"arbitrary_types_allowed": True}

    role: str = Field(..., description="'user', 'model', or 'tool'")
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_name: Optional[str] = None
    tool_result: Optional[str] = None
    raw_content: Optional[Any] = None  # Preserves SDK Content object (thought signatures, etc.)


class LLMResponse(BaseModel):
    """Standardized response from an LLM call."""

    model_config = {"arbitrary_types_allowed": True}

    text: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    raw_response: Any = None
    raw_content: Any = None


class BaseLLMClient(ABC):
    """Abstract base class for swappable LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Tool]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Send messages and available tools to the LLM and return its response."""
        pass


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini client utilizing native function calling via the official google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        retry_limit: int = 4,
        backoff_factor: float = 2.0,
        http_timeout: float = 60.0,
        logger: Optional[AgentLogger] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if genai is None:
            raise ImportError(
                "google-genai SDK is not installed. Please install it with `pip install google-genai`."
            )
        self.api_key = api_key
        self.model_name = model_name
        self.retry_limit = retry_limit
        self.backoff_factor = backoff_factor
        self.http_timeout = http_timeout
        self.logger = logger
        self.sleep_fn = sleep_fn

        # google-genai HttpOptions.timeout expects milliseconds
        self.timeout_ms = int(http_timeout * 1000) if http_timeout < 1000 else int(http_timeout)
        http_opts = types.HttpOptions(timeout=self.timeout_ms) if types else None

        # Initialize the official SDK client
        if api_key:
            self.client = genai.Client(api_key=api_key, http_options=http_opts)
        else:
            self.client = genai.Client(http_options=http_opts)

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Any]:
        """Convert provider-agnostic ChatMessages to google-genai types.Content objects.

        Rules:
        - SDK requires roles 'user' and 'model' only.
        - Internal 'tool' messages are converted to types.Content(role='user', parts=[...]).
        - Multiple consecutive tool messages are merged into a SINGLE user Content in order.
        - Preserved model Content objects (raw_content) are forwarded unchanged to keep thought signatures.
        """
        sdk_contents: List[Any] = []
        i = 0
        n = len(messages)

        while i < n:
            msg = messages[i]

            if msg.role == "user":
                part = types.Part.from_text(text=msg.content or "")
                sdk_contents.append(types.Content(role="user", parts=[part]))
                i += 1

            elif msg.role == "model":
                if msg.raw_content is not None:
                    sdk_contents.append(msg.raw_content)
                else:
                    parts = []
                    if msg.content:
                        parts.append(types.Part.from_text(text=msg.content))
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=tc.name,
                                        args=tc.arguments,
                                    )
                                )
                            )
                    sdk_contents.append(types.Content(role="model", parts=parts))
                i += 1

            elif msg.role == "tool":
                # Collect ALL consecutive tool messages into a single user Content
                tool_parts = []
                while i < n and messages[i].role == "tool":
                    t_msg = messages[i]
                    response_payload = {"result": t_msg.tool_result or t_msg.content or ""}
                    part = types.Part.from_function_response(
                        name=t_msg.tool_name or "unknown_tool",
                        response=response_payload,
                    )
                    tool_parts.append(part)
                    i += 1
                sdk_contents.append(types.Content(role="user", parts=tool_parts))

            else:
                i += 1

        return sdk_contents

    def _convert_tools(self, tools: Optional[List[Tool]]) -> Optional[List[Any]]:
        """Convert Tool instances to google-genai types.Tool function declarations."""
        if not tools:
            return None

        declarations = []
        for tool in tools:
            decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool.get_json_schema(),
            )
            declarations.append(decl)

        return [types.Tool(function_declarations=declarations)]

    def _extract_status_code(self, err: Exception) -> Optional[int]:
        """Extract HTTP status code from an exception if available."""
        code = getattr(err, "code", None)
        if isinstance(code, int):
            return code
        resp = getattr(err, "response", None)
        if resp and hasattr(resp, "status_code") and isinstance(resp.status_code, int):
            return resp.status_code
        match = re.search(r"\b(400|401|403|404|429|500|503)\b", str(err))
        if match:
            return int(match.group(1))
        return None

    def _compute_retry_delay(self, err: Exception, attempt: int) -> float:
        """Compute retry delay honoring server guidance or applying capped exponential backoff."""
        # 1. Check HTTP response headers
        resp = getattr(err, "response", None)
        if resp and hasattr(resp, "headers") and resp.headers:
            retry_after = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), 60.0)
                except ValueError:
                    pass

        # 2. Check error message for suggested delay
        match = re.search(
            r"retry(?:\s+after|\s*delay)?[:\s]+(\d+(?:\.\d+)?)\s*s?",
            str(err),
            re.IGNORECASE,
        )
        if match:
            try:
                return min(float(match.group(1)), 60.0)
            except ValueError:
                pass

        # 3. Exponential backoff with jitter, capped at 30 seconds
        delay = (self.backoff_factor ** attempt) + random.uniform(0.5, 1.5)
        return min(delay, 30.0)

    def generate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Tool]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Call Gemini model with strict status-code filtering and retry logic."""
        contents = self._convert_messages(messages)
        gemini_tools = self._convert_tools(tools)

        config_kwargs: Dict[str, Any] = {}
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools
            # Disable automatic execution so the agent loop retains full control
            config_kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if self.http_timeout:
            config_kwargs["http_options"] = types.HttpOptions(timeout=self.timeout_ms)

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        last_exception: Optional[Exception] = None
        for attempt in range(1, self.retry_limit + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

                candidate = response.candidates[0] if (response and response.candidates) else None
                raw_content = candidate.content if candidate else None

                text_parts: List[str] = []
                tool_calls: List[ToolCall] = []

                if candidate and candidate.content and candidate.content.parts:
                    for idx, part in enumerate(candidate.content.parts):
                        # Extract text from text parts (avoiding response.text warning on function call parts)
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)
                        # Extract function call requests
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            tool_calls.append(
                                ToolCall(
                                    id=f"call_{idx}_{fc.name}",
                                    name=fc.name,
                                    arguments=dict(fc.args or {}),
                                )
                            )

                text_content = "\n".join(text_parts) if text_parts else None

                return LLMResponse(
                    text=text_content,
                    tool_calls=tool_calls,
                    raw_response=response,
                    raw_content=raw_content,
                )

            except Exception as e:
                last_exception = e
                code = self._extract_status_code(e)

                # Rule 1: Fail immediately on 400, 401, 403, 404 (non-retryable client errors)
                if code in (400, 401, 403, 404):
                    raise RuntimeError(
                        f"Non-retryable client error ({code}): {str(e)}. "
                        f"Check request parameters, authentication, or model name."
                    ) from e

                # Rule 2: Determine if error is retryable (429, 500, 503, timeouts, connection errors)
                is_retryable = False
                if code in (429, 500, 503):
                    is_retryable = True
                elif isinstance(e, (TimeoutError, ConnectionError)):
                    is_retryable = True
                else:
                    err_str = str(e).lower()
                    if any(
                        ph in err_str
                        for ph in (
                            "timeout",
                            "timed out",
                            "connection error",
                            "temporarily unavailable",
                            "resource_exhausted",
                            "rate limit",
                            "quota",
                        )
                    ):
                        is_retryable = True

                if is_retryable and attempt < self.retry_limit:
                    delay = self._compute_retry_delay(e, attempt)
                    if self.logger:
                        self.logger.log_retry(
                            attempt=attempt,
                            max_retries=self.retry_limit,
                            delay_s=delay,
                            error_msg=str(e),
                        )
                    self.sleep_fn(delay)
                else:
                    break

        raise RuntimeError(
            f"Gemini API request failed after {self.retry_limit} attempt(s): {str(last_exception)}"
        ) from last_exception


class MockLLMClient(BaseLLMClient):
    """Mock LLM Client for hermetic testing without external API calls."""

    def __init__(
        self,
        responses: Optional[List[LLMResponse]] = None,
        response_handler: Optional[Callable[[List[ChatMessage]], LLMResponse]] = None,
    ):
        self.responses = list(responses or [])
        self.response_handler = response_handler
        self.recorded_history: List[List[ChatMessage]] = []

    def generate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Tool]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        self.recorded_history.append(list(messages))

        if self.response_handler:
            return self.response_handler(messages)

        if self.responses:
            return self.responses.pop(0)

        return LLMResponse(text="Mock LLM final answer: Goal achieved.", tool_calls=[])
