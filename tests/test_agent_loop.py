"""Integration tests for the Agent execution loop using MockLLMClient."""

from pathlib import Path
from typing import Any, Dict, List
import pytest

from agent.agent import Agent, AgentResult
from agent.config import Settings
from agent.llm_client import ChatMessage, LLMResponse, MockLLMClient, ToolCall
from agent.logger import AgentLogger
from agent.tools.registry import ToolRegistry


def test_agent_multi_step_workflow(temp_workspace: Path, silent_logger: AgentLogger):
    """Test a 3-step workflow: calculate -> write file -> complete with final answer."""
    settings = Settings(
        max_iterations=5,
        workspace_dir=temp_workspace,
        approve_risky=False,
        verbose=False,
    )
    registry = ToolRegistry.create_default(workspace_dir=temp_workspace)

    # Sequence of mock LLM turns
    responses = [
        # Turn 1: LLM decides to calculate 25 * 4
        LLMResponse(
            text="Let me calculate the total value first.",
            tool_calls=[ToolCall(name="calculator", arguments={"expression": "25 * 4"})],
        ),
        # Turn 2: LLM writes the result to a report
        LLMResponse(
            text="Now I will save the result to report.txt.",
            tool_calls=[
                ToolCall(
                    name="file_write",
                    arguments={"file_path": "report.txt", "content": "Total: 100"},
                )
            ],
        ),
        # Turn 3: LLM provides final answer
        LLMResponse(
            text="Task completed! Calculated 100 and saved to report.txt.",
            tool_calls=[],
        ),
    ]

    mock_llm = MockLLMClient(responses=responses)
    agent = Agent(llm_client=mock_llm, registry=registry, settings=settings, logger=silent_logger)

    result = agent.run("Calculate 25*4 and write to report.txt")

    assert result.success is True
    assert result.iterations == 3
    assert "Task completed!" in result.final_answer
    assert (temp_workspace / "report.txt").exists()
    assert (temp_workspace / "report.txt").read_text(encoding="utf-8") == "Total: 100"


def test_agent_self_correction_on_validation_error(
    temp_workspace: Path, silent_logger: AgentLogger
):
    """Test that LLM receives validation errors and successfully self-corrects."""
    settings = Settings(max_iterations=5, workspace_dir=temp_workspace)
    registry = ToolRegistry.create_default(workspace_dir=temp_workspace)

    def dynamic_llm_handler(messages: List[ChatMessage]) -> LLMResponse:
        # Inspect the latest message
        latest = messages[-1]
        if latest.role == "user":
            # First turn: intentionally send invalid arguments (missing 'expression')
            return LLMResponse(
                text="Calling calculator with wrong arg...",
                tool_calls=[ToolCall(name="calculator", arguments={"bad_param": "42"})],
            )
        elif latest.role == "tool" and "Validation Error" in (latest.tool_result or ""):
            # Self-correcting turn: fix argument to 'expression'
            return LLMResponse(
                text="Correcting arguments based on error.",
                tool_calls=[ToolCall(name="calculator", arguments={"expression": "42 * 2"})],
            )
        else:
            # Final turn
            return LLMResponse(text="Calculation finished: 84.", tool_calls=[])

    mock_llm = MockLLMClient(response_handler=dynamic_llm_handler)
    agent = Agent(llm_client=mock_llm, registry=registry, settings=settings, logger=silent_logger)

    result = agent.run("Calculate 42 * 2")

    assert result.success is True
    assert "84" in result.final_answer
    # History must show the validation error turn and recovery
    tool_observations = [m.tool_result for m in result.history if m.role == "tool"]
    assert any("Validation Error" in obs for obs in tool_observations if obs)
    assert any("Result: 84" in obs for obs in tool_observations if obs)


def test_agent_loop_guard(temp_workspace: Path, silent_logger: AgentLogger):
    """Test that calling the exact same tool with exact same arguments 3 times trips the loop guard."""
    settings = Settings(max_iterations=5, workspace_dir=temp_workspace)
    registry = ToolRegistry.create_default(workspace_dir=temp_workspace)

    # 3 identical calls followed by final answer
    responses = [
        LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expression": "1 + 1"})]),
        LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expression": "1 + 1"})]),
        LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expression": "1 + 1"})]),
        LLMResponse(text="I see the loop guard warning. The answer is 2.", tool_calls=[]),
    ]

    mock_llm = MockLLMClient(responses=responses)
    agent = Agent(llm_client=mock_llm, registry=registry, settings=settings, logger=silent_logger)

    result = agent.run("Add 1 + 1")

    assert result.success is True
    tool_observations = [m.tool_result for m in result.history if m.role == "tool"]
    assert any("Loop Guard Triggered" in obs for obs in tool_observations if obs)


def test_agent_max_iterations_limit(temp_workspace: Path, silent_logger: AgentLogger):
    """Test that agent halts gracefully when max_iterations is reached."""
    settings = Settings(max_iterations=2, workspace_dir=temp_workspace)
    registry = ToolRegistry.create_default(workspace_dir=temp_workspace)

    # Infinite tool calling simulation
    responses = [
        LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expression": "1 + 1"})]),
        LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expression": "2 + 2"})]),
        LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expression": "3 + 3"})]),
    ]

    mock_llm = MockLLMClient(responses=responses)
    agent = Agent(llm_client=mock_llm, registry=registry, settings=settings, logger=silent_logger)

    result = agent.run("Keep calculating")

    assert result.success is False
    assert result.iterations == 2
    assert "Reached maximum iteration limit" in result.final_answer


def test_agent_risky_tool_approval(temp_workspace: Path, silent_logger: AgentLogger):
    """Test human-in-the-loop approval: denial prevents execution, approval allows execution."""
    settings = Settings(
        max_iterations=4,
        workspace_dir=temp_workspace,
        approve_risky=True,
    )
    registry = ToolRegistry.create_default(workspace_dir=temp_workspace)

    # Scenario 1: User denies permission
    mock_responses_denied = [
        LLMResponse(
            tool_calls=[
                ToolCall(
                    name="file_write",
                    arguments={"file_path": "secret.txt", "content": "data"},
                )
            ]
        ),
        LLMResponse(text="Permission was denied, halting.", tool_calls=[]),
    ]

    agent_denied = Agent(
        llm_client=MockLLMClient(responses=mock_responses_denied),
        registry=registry,
        settings=settings,
        logger=silent_logger,
        approval_callback=lambda tool, args: False,  # DENIED
    )

    result_denied = agent_denied.run("Write to secret.txt")
    assert not (temp_workspace / "secret.txt").exists()
    tool_observations = [m.tool_result for m in result_denied.history if m.role == "tool"]
    assert any("User declined approval" in obs for obs in tool_observations if obs)

    # Scenario 2: User approves permission
    mock_responses_approved = [
        LLMResponse(
            tool_calls=[
                ToolCall(
                    name="file_write",
                    arguments={"file_path": "approved.txt", "content": "authorized data"},
                )
            ]
        ),
        LLMResponse(text="File written successfully.", tool_calls=[]),
    ]

    agent_approved = Agent(
        llm_client=MockLLMClient(responses=mock_responses_approved),
        registry=registry,
        settings=settings,
        logger=silent_logger,
        approval_callback=lambda tool, args: True,  # APPROVED
    )

    result_approved = agent_approved.run("Write to approved.txt")
    assert (temp_workspace / "approved.txt").exists()
    assert (temp_workspace / "approved.txt").read_text(encoding="utf-8") == "authorized data"
