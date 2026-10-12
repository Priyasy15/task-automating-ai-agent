"""Unit tests for individual tools, sandbox security, validation, and timeouts."""

import os
import time
from pathlib import Path
import pytest
from pydantic import BaseModel, Field

from agent.tools.base import Tool
from agent.tools.calculator import CalculatorTool
from agent.tools.datetime_tool import DateTimeTool
from agent.tools.file_ops import FileListTool, FileReadTool, FileWriteTool
from agent.tools.summarizer import SummarizerTool


# ---------------------------------------------------------------------
# Calculator Tests & DoS Protections
# ---------------------------------------------------------------------

def test_calculator_basic_arithmetic():
    calc = CalculatorTool()
    res = calc.execute(expression="25 * 4 + 10")
    assert res == "Result: 110"


def test_calculator_functions_and_constants():
    calc = CalculatorTool()
    res = calc.execute(expression="sqrt(144) + round(3.7)")
    assert res == "Result: 16"


def test_calculator_division_by_zero():
    calc = CalculatorTool()
    res = calc.execute(expression="100 / 0")
    assert "ZeroDivisionError" in res or "Division by zero" in res


def test_calculator_syntax_error():
    calc = CalculatorTool()
    res = calc.execute(expression="25 * + / 10")
    assert "Calculation Error" in res


def test_calculator_blocks_code_execution():
    calc = CalculatorTool()
    res = calc.execute(expression="__import__('os').system('dir')")
    assert "Calculation Error" in res
    assert "Unauthorized or unknown function" in res


def test_calculator_dos_huge_exponent():
    """Huge exponents (e.g. 9**9**9 or 9**999) must be capped before CPU exhaustion."""
    calc = CalculatorTool()
    res_chained = calc.execute(expression="9**9**9")
    assert "Calculation Error" in res_chained
    assert "exceeds safe limit" in res_chained

    res_huge = calc.execute(expression="2**500")
    assert "Calculation Error" in res_huge
    assert "exceeds safe limit" in res_huge


def test_calculator_dos_huge_operand():
    """Massive number literals must be capped to prevent memory allocation DoS."""
    calc = CalculatorTool()
    res = calc.execute(expression="99999999999999999999999999999999 + 1")
    assert "Calculation Error" in res
    assert "exceeds safe limit" in res


def test_calculator_dos_expression_length():
    """Overly long expressions must be rejected upfront."""
    calc = CalculatorTool()
    long_expr = "1 + " * 100 + "1"  # > 400 characters
    res = calc.execute(expression=long_expr)
    assert "Calculation Error" in res
    assert "exceeds maximum length" in res


# ---------------------------------------------------------------------
# File Operations & Sandboxing Tests
# ---------------------------------------------------------------------

def test_file_write_and_read(temp_workspace: Path):
    writer = FileWriteTool(workspace_dir=temp_workspace)
    reader = FileReadTool(workspace_dir=temp_workspace)

    # Write file
    write_res = writer.execute(file_path="subfolder/test.txt", content="Hello Sandbox!")
    assert "Success" in write_res
    assert (temp_workspace / "subfolder" / "test.txt").exists()

    # Read file
    read_res = reader.execute(file_path="subfolder/test.txt")
    assert "Hello Sandbox!" in read_res


def test_file_write_append(temp_workspace: Path):
    writer = FileWriteTool(workspace_dir=temp_workspace)
    reader = FileReadTool(workspace_dir=temp_workspace)

    writer.execute(file_path="log.txt", content="Line 1\n")
    writer.execute(file_path="log.txt", content="Line 2\n", append=True)

    read_res = reader.execute(file_path="log.txt")
    assert "Line 1\nLine 2" in read_res


def test_file_list(temp_workspace: Path):
    writer = FileWriteTool(workspace_dir=temp_workspace)
    lister = FileListTool(workspace_dir=temp_workspace)

    writer.execute(file_path="file_a.txt", content="A")
    writer.execute(file_path="file_b.txt", content="B")

    list_res = lister.execute()
    assert "file_a.txt" in list_res
    assert "file_b.txt" in list_res


def test_sandbox_blocks_dot_dot_traversal(temp_workspace: Path):
    """Path traversal via '..' must be rejected."""
    writer = FileWriteTool(workspace_dir=temp_workspace)
    reader = FileReadTool(workspace_dir=temp_workspace)

    write_res = writer.execute(file_path="../../evil.txt", content="malicious")
    assert "Security Violation" in write_res or "Security Error" in write_res

    read_res = reader.execute(file_path="../evil.txt")
    assert "Security Violation" in read_res or "Security Error" in read_res


def test_sandbox_blocks_absolute_path_escape(temp_workspace: Path):
    """Absolute paths outside the workspace must be rejected."""
    writer = FileWriteTool(workspace_dir=temp_workspace)
    outside_path = temp_workspace.parent / "outside.txt"
    write_res = writer.execute(file_path=str(outside_path), content="escape")
    assert "Security Violation" in write_res or "Security Error" in write_res


def test_sandbox_blocks_sibling_directory_with_similar_name(temp_workspace: Path):
    """A sibling directory starting with the same prefix (e.g. workspace_evil) must be blocked."""
    sibling_dir = temp_workspace.parent / f"{temp_workspace.name}_evil"
    sibling_file = sibling_dir / "target.txt"

    writer = FileWriteTool(workspace_dir=temp_workspace)
    write_res = writer.execute(file_path=str(sibling_file), content="escape")
    assert "Security Violation" in write_res or "Security Error" in write_res


def test_sandbox_blocks_symlink_escape(temp_workspace: Path):
    """Symlinks inside the workspace pointing outside must be caught by resolution."""
    outside_target = temp_workspace.parent / "secret_outside.txt"
    outside_target.write_text("classified", encoding="utf-8")
    symlink_path = temp_workspace / "symlink_to_outside.txt"

    try:
        os.symlink(str(outside_target), str(symlink_path))
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("Symlink creation not supported or requires elevated privileges on this OS.")

    reader = FileReadTool(workspace_dir=temp_workspace)
    read_res = reader.execute(file_path="symlink_to_outside.txt")
    assert "Security Violation" in read_res or "Security Error" in read_res


# ---------------------------------------------------------------------
# DateTime & Summarizer Tests
# ---------------------------------------------------------------------

def test_datetime_tool():
    dt_tool = DateTimeTool()
    res = dt_tool.execute()
    assert "Local Time:" in res
    assert "UTC Time:" in res


def test_summarizer_tool():
    summarizer = SummarizerTool()
    long_text = (
        "Python 3.13 was released with major improvements. "
        "It features an experimental free-threaded build without the GIL. "
        "A new interactive interpreter is based on PyPy code. "
        "Performance optimizations improve execution speed noticeably across workloads. "
        "Developers are encouraged to test their libraries against the new release."
    )
    res = summarizer.execute(text=long_text, max_bullet_points=3, focus="free-threaded")
    assert "Summary" in res
    assert "•" in res


# ---------------------------------------------------------------------
# Base Tool: Validation, Prompt Timeout, and Truncation Tests
# ---------------------------------------------------------------------

class DummySlowArgs(BaseModel):
    sleep_duration: float = Field(..., description="Duration to sleep")


class DummySlowTool(Tool):
    name = "dummy_slow"
    description = "A tool that sleeps to test timeouts"
    args_schema = DummySlowArgs

    def execute(self, sleep_duration: float, **kwargs) -> str:
        time.sleep(sleep_duration)
        return "Finished sleeping"


def test_tool_timeout_enforcement_returns_promptly():
    """Tool that sleeps 1.5s with a 0.1s timeout must return promptly without blocking."""
    tool = DummySlowTool()
    start_t = time.time()
    res = tool.run_with_timeout(args={"sleep_duration": 1.5}, timeout_seconds=0.1)
    duration = time.time() - start_t

    assert "Timeout Error" in res
    assert "timed out after 0.1 seconds" in res
    # Must have returned promptly near the 0.1s timeout, well before the 1.5s sleep completes
    assert duration < 0.5, f"Execution took {duration}s, expected < 0.5s"


def test_tool_validation_error_feedback():
    calc = CalculatorTool()
    res = calc.run_with_timeout(args={})
    assert "Validation Error" in res
    assert "expression" in res


def test_tool_output_truncation():
    class DummyLongTool(Tool):
        name = "dummy_long"
        description = "Returns long text"
        args_schema = CalculatorTool.args_schema

        def execute(self, expression: str, **kwargs) -> str:
            return "X" * 1000

    tool = DummyLongTool()
    res = tool.run_with_timeout(args={"expression": "test"}, max_output_length=200)
    assert len(res) < 350
    assert "Output truncated" in res
