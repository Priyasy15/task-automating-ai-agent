"""Pytest configuration and shared test fixtures."""

import shutil
import tempfile
from pathlib import Path
from typing import Generator
import pytest

from agent.config import Settings
from agent.logger import AgentLogger


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Provide a temporary directory simulating the sandboxed workspace."""
    temp_dir = tempfile.mkdtemp(prefix="agent_test_ws_")
    ws_path = Path(temp_dir).resolve()
    yield ws_path
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_settings(temp_workspace: Path) -> Settings:
    """Provide test settings pointing to the temporary workspace."""
    return Settings(
        gemini_api_key="test_mock_key",
        gemini_model="gemini-test",
        max_iterations=5,
        tool_timeout_seconds=2.0,
        max_output_length=500,
        retry_limit=2,
        backoff_factor=1.1,
        workspace_dir=temp_workspace,
        approve_risky=False,
        verbose=False,
    )


@pytest.fixture
def silent_logger(temp_workspace: Path) -> AgentLogger:
    """Provide an AgentLogger that writes to a temporary logs directory without console spam."""
    logs_dir = temp_workspace / "test_logs"
    return AgentLogger(logs_dir=str(logs_dir), verbose=False)
