"""Unit tests for the LLM client abstraction, role conversion, and retry logic."""

from unittest.mock import MagicMock
import pytest

from agent.llm_client import ChatMessage, GeminiLLMClient, ToolCall
from agent.tools.calculator import CalculatorTool


def test_gemini_message_roles_are_only_user_or_model():
    """Gemini API only supports roles 'user' and 'model'. Internal 'tool' becomes 'user'."""
    client = GeminiLLMClient(api_key="dummy_key", model_name="gemini-test")

    messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(
            role="model",
            content="Thinking",
            tool_calls=[ToolCall(name="calculator", arguments={"expression": "2+2"})],
        ),
        ChatMessage(
            role="tool",
            tool_name="calculator",
            tool_result="Result: 4",
        ),
    ]

    sdk_contents = client._convert_messages(messages)
    assert len(sdk_contents) == 3
    # Every message sent to SDK must have role 'user' or 'model' only
    for content in sdk_contents:
        assert content.role in ("user", "model")

    assert sdk_contents[0].role == "user"
    assert sdk_contents[1].role == "model"
    assert sdk_contents[2].role == "user"
    # The tool response part must be present in the user content
    assert len(sdk_contents[2].parts) == 1
    assert sdk_contents[2].parts[0].function_response.name == "calculator"
    assert sdk_contents[2].parts[0].function_response.response == {"result": "Result: 4"}


def test_consecutive_tool_results_merged_into_single_user_content():
    """Multiple tool results from a single turn must merge into ONE user Content in order."""
    client = GeminiLLMClient(api_key="dummy_key", model_name="gemini-test")

    messages = [
        ChatMessage(role="user", content="Calculate both"),
        ChatMessage(
            role="model",
            content="Running tools in parallel",
            tool_calls=[
                ToolCall(name="calc_a", arguments={"x": 1}),
                ToolCall(name="calc_b", arguments={"x": 2}),
            ],
        ),
        ChatMessage(role="tool", tool_name="calc_a", tool_result="Result A"),
        ChatMessage(role="tool", tool_name="calc_b", tool_result="Result B"),
    ]

    sdk_contents = client._convert_messages(messages)
    # user turn 1, model turn, merged user turn with 2 function responses
    assert len(sdk_contents) == 3
    merged_user_content = sdk_contents[2]
    assert merged_user_content.role == "user"
    assert len(merged_user_content.parts) == 2
    assert merged_user_content.parts[0].function_response.name == "calc_a"
    assert merged_user_content.parts[0].function_response.response == {"result": "Result A"}
    assert merged_user_content.parts[1].function_response.name == "calc_b"
    assert merged_user_content.parts[1].function_response.response == {"result": "Result B"}


def test_preserve_raw_content_for_model_turn():
    """Model turns with preserved raw_content should be forwarded unchanged."""
    client = GeminiLLMClient(api_key="dummy_key", model_name="gemini-test")

    mock_raw = MagicMock()
    mock_raw.role = "model"

    messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="model", content="Test", raw_content=mock_raw),
    ]

    sdk_contents = client._convert_messages(messages)
    assert len(sdk_contents) == 2
    assert sdk_contents[1] is mock_raw


def test_gemini_tool_conversion():
    client = GeminiLLMClient(api_key="dummy_key", model_name="gemini-test")
    tools = [CalculatorTool()]

    sdk_tools = client._convert_tools(tools)
    assert sdk_tools is not None
    assert len(sdk_tools) == 1
    assert len(sdk_tools[0].function_declarations) == 1
    assert sdk_tools[0].function_declarations[0].name == "calculator"


def test_gemini_400_is_not_retried():
    """HTTP 400 Bad Request must fail immediately without retrying."""
    sleep_calls = []
    client = GeminiLLMClient(
        api_key="dummy_key",
        model_name="gemini-test",
        retry_limit=4,
        sleep_fn=lambda s: sleep_calls.append(s),
    )

    err = Exception("400 Role 'tool' is not supported")
    client.client.models.generate_content = MagicMock(side_effect=err)

    messages = [ChatMessage(role="user", content="Hi")]
    with pytest.raises(RuntimeError) as exc_info:
        client.generate(messages=messages)

    assert "Non-retryable client error (400)" in str(exc_info.value)
    # Must have only been called once
    assert client.client.models.generate_content.call_count == 1
    assert len(sleep_calls) == 0


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_gemini_retry_on_retryable_codes(status_code):
    """HTTP 429, 500, 503 must retry with backoff, using injected sleep_fn."""
    sleep_calls = []
    client = GeminiLLMClient(
        api_key="dummy_key",
        model_name="gemini-test",
        retry_limit=3,
        backoff_factor=1.5,
        sleep_fn=lambda s: sleep_calls.append(s),
    )

    mock_candidate = MagicMock()
    mock_part = MagicMock()
    mock_part.text = "Success after retry"
    mock_part.function_call = None
    mock_candidate.content.parts = [mock_part]

    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]

    err = Exception(f"{status_code} Server/Rate error")
    client.client.models.generate_content = MagicMock(side_effect=[err, mock_response])

    messages = [ChatMessage(role="user", content="Hi")]
    response = client.generate(messages=messages)

    assert response.text == "Success after retry"
    assert client.client.models.generate_content.call_count == 2
    assert len(sleep_calls) == 1
    assert sleep_calls[0] > 0


def test_gemini_honors_suggested_retry_delay():
    """On 429, server suggested retry delay should be respected."""
    sleep_calls = []
    client = GeminiLLMClient(
        api_key="dummy_key",
        model_name="gemini-test",
        retry_limit=3,
        sleep_fn=lambda s: sleep_calls.append(s),
    )

    mock_candidate = MagicMock()
    mock_part = MagicMock()
    mock_part.text = "OK"
    mock_part.function_call = None
    mock_candidate.content.parts = [mock_part]

    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]

    err = Exception("429 Resource Exhausted. Please retry after: 4.5s")
    client.client.models.generate_content = MagicMock(side_effect=[err, mock_response])

    messages = [ChatMessage(role="user", content="Hi")]
    client.generate(messages=messages)

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == 4.5
