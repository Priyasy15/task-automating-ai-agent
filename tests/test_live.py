"""Optional live integration test against Gemini API (run with pytest -m live)."""

import os
import pytest
from dotenv import load_dotenv

from agent.llm_client import ChatMessage, GeminiLLMClient

load_dotenv()


@pytest.mark.live
def test_single_live_gemini_call():
    """Single tiny live API call to verify connection and model response."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key in ("your_key_here", "your_gemini_api_key_here"):
        pytest.skip("GEMINI_API_KEY not configured for live test.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = GeminiLLMClient(
        api_key=api_key,
        model_name=model_name,
        retry_limit=2,
    )

    response = client.generate(
        messages=[ChatMessage(role="user", content="Reply with only the word: PONG")],
        tools=None,
    )

    assert response.text is not None
    assert len(response.text.strip()) > 0
