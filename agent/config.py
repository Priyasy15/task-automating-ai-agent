"""Configuration management for the AI Agent."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file if present
load_dotenv()


class Settings(BaseModel):
    """Global configuration settings for the AI Agent.

    Attributes:
        gemini_api_key: API key for Google Gemini (reads from GEMINI_API_KEY).
        gemini_model: Model name for Gemini (reads from GEMINI_MODEL).
        max_iterations: Maximum loop turns before stopping to prevent runaway loops.
        tool_timeout_seconds: Maximum execution time allowed for a single tool call.
        max_output_length: Maximum character length of tool observations before truncation.
        retry_limit: Max retry attempts on transient network or 429 rate limit errors.
        backoff_factor: Base multiplier for exponential backoff (delay = backoff_factor ** attempt).
        workspace_dir: Sandboxed directory for all file operations.
        approve_risky: Whether human approval is required for risky tool executions.
        verbose: Whether detailed step execution is printed to console.
    """

    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    max_iterations: int = Field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "10"))
    )
    tool_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("TOOL_TIMEOUT_SECONDS", "20.0"))
    )
    max_output_length: int = Field(
        default_factory=lambda: int(os.getenv("MAX_OUTPUT_LENGTH", "2000"))
    )
    retry_limit: int = Field(
        default_factory=lambda: int(os.getenv("RETRY_LIMIT", "4"))
    )
    backoff_factor: float = Field(
        default_factory=lambda: float(os.getenv("BACKOFF_FACTOR", "2.0"))
    )
    workspace_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("WORKSPACE_DIR", "./workspace")).resolve()
    )
    approve_risky: bool = False
    verbose: bool = False

    def validate_api_key(self) -> None:
        """Validate that the Gemini API key is configured."""
        if not self.gemini_api_key or self.gemini_api_key in ("your_key_here", "your_gemini_api_key_here"):
            raise ValueError(
                "GEMINI_API_KEY is not set. Please obtain a free API key from "
                "https://aistudio.google.com/app/apikey and add it to your .env file."
            )

    @classmethod
    def load(
        cls,
        model: Optional[str] = None,
        max_iterations: Optional[int] = None,
        approve_risky: Optional[bool] = None,
        verbose: Optional[bool] = None,
    ) -> "Settings":
        """Instantiate settings with optional overrides from CLI arguments."""
        settings = cls()
        if model:
            settings.gemini_model = model
        if max_iterations is not None:
            settings.max_iterations = max_iterations
        if approve_risky is not None:
            settings.approve_risky = approve_risky
        if verbose is not None:
            settings.verbose = verbose

        # Ensure workspace directory exists
        settings.workspace_dir.mkdir(parents=True, exist_ok=True)
        return settings
