"""Core AI Agent loop implementing the ReAct pattern."""

import json
import time
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from agent.config import Settings
from agent.llm_client import BaseLLMClient, ChatMessage, LLMResponse, ToolCall
from agent.logger import AgentLogger
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry


class AgentResult(BaseModel):
    """Encapsulates the final outcome of an agent execution run."""

    goal: str
    final_answer: str
    success: bool
    iterations: int
    total_duration_s: float
    history: List[ChatMessage] = Field(default_factory=list)


DEFAULT_SYSTEM_INSTRUCTION = """You are an expert autonomous task-automating AI agent.
Your mission is to accomplish the user's goal by deciding the best sequence of tools to call.

Follow these strict operating principles:
1. THINK BEFORE ACTING: Analyze the goal, break it down into sequential steps, and execute one step at a time.
2. USE TOOLS EFFECTIVELY: When you need facts, files, or calculations, invoke the appropriate tool with precise arguments.
3. GROUNDED WORKFLOW: Use observations from previous tool executions to guide your next decision. Never guess or hallucinate tool results.
4. STOPPING CRITERIA: When you have completed all required actions (e.g., searched, summarized, and saved a file), provide a comprehensive, clear final answer to the user WITHOUT invoking any further tools.
"""


class Agent:
    """The task-automating agent managing the execution loop and error controls."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        registry: ToolRegistry,
        settings: Settings,
        logger: Optional[AgentLogger] = None,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        approval_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ):
        self.llm_client = llm_client
        self.registry = registry
        self.settings = settings
        self.logger = logger or AgentLogger(verbose=settings.verbose)
        self.system_instruction = system_instruction
        self.approval_callback = approval_callback

    def _request_user_approval(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Request user confirmation before running a risky tool."""
        if self.approval_callback is not None:
            return self.approval_callback(tool_name, args)

        print("\n" + "!" * 60)
        print(f"⚠️  [APPROVAL REQUIRED] Tool '{tool_name}' is flagged as RISKY.")
        print(f"Arguments: {json.dumps(args, indent=2, ensure_ascii=False)}")
        print("!" * 60)
        try:
            choice = input(f"Authorize '{tool_name}' execution? [y/N]: ").strip().lower()
            return choice in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def _serialize_tool_call(self, name: str, args: Dict[str, Any]) -> str:
        """Create a canonical string signature for detecting repeated identical calls."""
        try:
            return f"{name}:{json.dumps(args, sort_keys=True)}"
        except Exception:
            return f"{name}:{str(args)}"

    def run(self, goal: str) -> AgentResult:
        """Execute the agent loop for the given natural language goal."""
        start_time = time.time()
        self.logger.log_goal(goal)

        # Initialize conversation history with user goal
        history: List[ChatMessage] = [
            ChatMessage(role="user", content=goal)
        ]

        available_tools = self.registry.list_tools()
        call_history_signatures: List[str] = []
        consecutive_identical_count = 0
        final_answer = ""
        success = False
        iteration = 0

        while iteration < self.settings.max_iterations:
            iteration += 1

            # Step 1: Call LLM with current history and tool declarations
            try:
                response: LLMResponse = self.llm_client.generate(
                    messages=history,
                    tools=available_tools,
                    system_instruction=self.system_instruction,
                )
            except Exception as e:
                error_msg = f"LLM Generation Error: {str(e)}"
                self.logger.log_warning(error_msg, iteration=iteration)
                final_answer = f"Agent halted due to an LLM error: {str(e)}"
                break

            # Step 2: Log reasoning / model thoughts if present
            if response.text and response.tool_calls:
                self.logger.log_thought(iteration, response.text)

            # Step 3: Check termination condition (No tool calls -> Final Answer)
            if not response.tool_calls:
                final_answer = response.text or "Goal completed with no text response."
                success = True
                duration = time.time() - start_time
                self.logger.log_final_answer(iteration, final_answer, duration)
                # Append final model response to history
                history.append(
                    ChatMessage(
                        role="model",
                        content=final_answer,
                        raw_content=response.raw_content,
                    )
                )
                break

            # Step 4: Record model's assistant turn (including tool calls) in history
            history.append(
                ChatMessage(
                    role="model",
                    content=response.text,
                    tool_calls=response.tool_calls,
                    raw_content=response.raw_content,
                )
            )

            # Step 5: Execute each requested tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call.name
                tool_args = tool_call.arguments
                tool: Optional[Tool] = self.registry.get_tool(tool_name)

                # Check Loop Guard (identical tool calls detector)
                signature = self._serialize_tool_call(tool_name, tool_args)
                if call_history_signatures and call_history_signatures[-1] == signature:
                    consecutive_identical_count += 1
                else:
                    consecutive_identical_count = 1
                call_history_signatures.append(signature)

                if consecutive_identical_count >= 3:
                    loop_msg = (
                        f"Loop Guard Triggered: You have invoked tool '{tool_name}' with identical "
                        f"arguments 3 times in a row without making progress. Stop repeating and "
                        f"provide your final answer based on existing information."
                    )
                    self.logger.log_warning(loop_msg, iteration=iteration)
                    history.append(
                        ChatMessage(
                            role="tool",
                            tool_name=tool_name,
                            tool_result=loop_msg,
                        )
                    )
                    continue

                if not tool:
                    unknown_msg = (
                        f"Error: Tool '{tool_name}' is not recognized or not allowed. "
                        f"Available tools are: {[t.name for t in available_tools]}."
                    )
                    self.logger.log_warning(unknown_msg, iteration=iteration)
                    history.append(
                        ChatMessage(
                            role="tool",
                            tool_name=tool_name,
                            tool_result=unknown_msg,
                        )
                    )
                    continue

                # Log tool invocation
                self.logger.log_tool_call(
                    iteration=iteration,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    is_risky=tool.is_risky,
                )

                # Step 6: Risky tool authorization check
                if tool.is_risky and self.settings.approve_risky:
                    authorized = self._request_user_approval(tool_name, tool_args)
                    if not authorized:
                        denied_msg = (
                            f"Action Cancelled: User declined approval for risky tool '{tool_name}'. "
                            f"Proceed without this tool action or propose an alternative."
                        )
                        self.logger.log_warning(denied_msg, iteration=iteration)
                        history.append(
                            ChatMessage(
                                role="tool",
                                tool_name=tool_name,
                                tool_result=denied_msg,
                            )
                        )
                        continue

                # Step 7: Execute tool safely with per-tool timeout and Pydantic validation
                tool_start = time.time()
                observation = tool.run_with_timeout(
                    args=tool_args,
                    timeout_seconds=self.settings.tool_timeout_seconds,
                    max_output_length=self.settings.max_output_length,
                )
                tool_duration_ms = (time.time() - tool_start) * 1000

                # Determine if execution resulted in error
                is_error = observation.startswith(("Validation Error", "Execution Error", "Security Error", "Timeout Error"))
                error_label = observation.split(":")[0] if is_error else None

                self.logger.log_tool_result(
                    iteration=iteration,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=observation,
                    duration_ms=tool_duration_ms,
                    error=error_label,
                )

                # Step 8: Feed observation back into conversation history
                history.append(
                    ChatMessage(
                        role="tool",
                        tool_name=tool_name,
                        tool_result=observation,
                    )
                )

        else:
            # Reached max_iterations without natural completion
            final_answer = (
                f"Agent halted: Reached maximum iteration limit of {self.settings.max_iterations} "
                f"without completing the goal."
            )
            success = False
            self.logger.log_warning(final_answer)

        total_duration = time.time() - start_time
        return AgentResult(
            goal=goal,
            final_answer=final_answer,
            success=success,
            iterations=iteration,
            total_duration_s=total_duration,
            history=history,
        )
