"""Step logging and tracing for the AI Agent.

Writes structured JSONL logs to `logs/run_<timestamp>.jsonl` and emits
formatted terminal output when verbose mode is active.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class AgentLogger:
    """Handles structured execution logging to console and JSONL files."""

    def __init__(self, logs_dir: str = "./logs", verbose: bool = False):
        self.logs_dir = Path(logs_dir)
        self.verbose = verbose
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"run_{timestamp}.jsonl"

    def _append_jsonl(self, payload: Dict[str, Any]) -> None:
        """Append a JSON record to the run log file."""
        payload["timestamp"] = datetime.now().isoformat()
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            if self.verbose:
                print(f"[Logger Error] Failed to write to {self.log_file}: {e}")

    def log_goal(self, goal: str) -> None:
        """Log the initial user goal."""
        record = {"step_type": "goal", "goal": goal}
        self._append_jsonl(record)
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"🎯 [GOAL] {goal}")
            print("=" * 60)

    def log_thought(self, iteration: int, thought: str) -> None:
        """Log model reasoning / thought text."""
        record = {
            "iteration": iteration,
            "step_type": "thought",
            "thought": thought,
        }
        self._append_jsonl(record)
        if self.verbose and thought.strip():
            print(f"\n🧠 [ITERATION {iteration}] Thinking:\n{thought.strip()}")

    def log_tool_call(
        self,
        iteration: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        is_risky: bool = False,
    ) -> None:
        """Log that a tool call was requested by the LLM."""
        record = {
            "iteration": iteration,
            "step_type": "tool_call",
            "tool": tool_name,
            "args": tool_args,
            "is_risky": is_risky,
        }
        self._append_jsonl(record)
        if self.verbose:
            risk_tag = " ⚠️ [RISKY]" if is_risky else ""
            print(f"🔧 [ACTION{risk_tag}] Tool: '{tool_name}'")
            print(f"   Args: {json.dumps(tool_args, ensure_ascii=False)}")

    def log_tool_result(
        self,
        iteration: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: str,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Log the result / observation of a tool execution."""
        record = {
            "iteration": iteration,
            "step_type": "observation",
            "tool": tool_name,
            "args": tool_args,
            "result": result,
            "duration_ms": round(duration_ms, 2),
            "error": error,
        }
        self._append_jsonl(record)
        if self.verbose:
            status_tag = "❌ [ERROR]" if error else "👁️ [OBSERVATION]"
            print(f"{status_tag} Tool '{tool_name}' ({round(duration_ms, 1)}ms):")
            # Truncate display preview if too long for console clarity
            preview = result if len(result) <= 300 else result[:300] + "... [truncated in console]"
            print(f"   {preview}")

    def log_retry(self, attempt: int, max_retries: int, delay_s: float, error_msg: str) -> None:
        """Log an API retry event (e.g. rate limit HTTP 429)."""
        record = {
            "step_type": "retry",
            "attempt": attempt,
            "max_retries": max_retries,
            "delay_s": delay_s,
            "error": error_msg,
        }
        self._append_jsonl(record)
        print(f"⏳ [RATE-LIMIT / RETRY] Attempt {attempt}/{max_retries} failed ({error_msg}). Retrying in {delay_s:.1f}s...")

    def log_warning(self, message: str, iteration: Optional[int] = None) -> None:
        """Log an execution warning (e.g. loop guard or validation error)."""
        record = {
            "iteration": iteration,
            "step_type": "warning",
            "message": message,
        }
        self._append_jsonl(record)
        if self.verbose:
            iter_prefix = f"[Iteration {iteration}] " if iteration else ""
            print(f"⚠️ [WARNING] {iter_prefix}{message}")

    def log_final_answer(self, iteration: int, answer: str, duration_s: float) -> None:
        """Log the agent's completed final answer."""
        record = {
            "iteration": iteration,
            "step_type": "final_answer",
            "answer": answer,
            "total_duration_s": round(duration_s, 2),
        }
        self._append_jsonl(record)
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"🏁 [FINAL ANSWER] Completed in {iteration} iteration(s) ({round(duration_s, 2)}s)")
            print("=" * 60)
            print(answer)
            print("=" * 60 + "\n")
