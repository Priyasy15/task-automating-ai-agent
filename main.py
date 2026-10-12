"""CLI entry point for the Task-Automating AI Agent."""

import argparse
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agent.config import Settings
from agent.logger import AgentLogger
from agent.tools.registry import ToolRegistry
from agent.llm_client import GeminiLLMClient
from agent.agent import Agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task-Automating AI Agent - Multi-step goal-oriented workflow automation."
    )
    parser.add_argument(
        "goal",
        type=str,
        help="Natural language goal for the agent (e.g. 'Search the latest Python release and save summary to notes.md')",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable real-time console tracing of thoughts, tool actions, and observations.",
    )
    parser.add_argument(
        "--approve-risky",
        action="store_true",
        help="Require manual user confirmation before executing risky tools (e.g. file_write).",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override Gemini model name (default from .env or 'gemini-2.5-flash').",
    )
    parser.add_argument(
        "--max-iterations", "-i",
        type=int,
        default=None,
        help="Override maximum allowed loop iterations (default 10).",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default=None,
        help="Comma-separated allowlist of enabled tools (e.g. 'calculator,file_write').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration and merge CLI flags
    settings = Settings.load(
        model=args.model,
        max_iterations=args.max_iterations,
        approve_risky=args.approve_risky,
        verbose=args.verbose,
    )

    # Check for GEMINI_API_KEY
    try:
        settings.validate_api_key()
    except ValueError as e:
        print("\n" + "=" * 65)
        print("❌ CONFIGURATION ERROR: GEMINI_API_KEY Missing")
        print("=" * 65)
        print(str(e))
        print("\nHow to fix:")
        print("1. Copy `.env.example` to `.env`:")
        print("   copy .env.example .env")
        print("2. Open `.env` and paste your key:")
        print("   GEMINI_API_KEY=AIzaSy...")
        print("3. Re-run your command.")
        print("=" * 65 + "\n")
        sys.exit(1)

    # Parse tool allowlist if provided
    allowed_tools = [t.strip() for t in args.tools.split(",")] if args.tools else None

    # Initialize components
    logger = AgentLogger(verbose=settings.verbose)
    registry = ToolRegistry.create_default(
        workspace_dir=settings.workspace_dir,
        allowed_tools=allowed_tools,
    )
    llm_client = GeminiLLMClient(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
        retry_limit=settings.retry_limit,
        backoff_factor=settings.backoff_factor,
        logger=logger,
    )
    agent = Agent(
        llm_client=llm_client,
        registry=registry,
        settings=settings,
        logger=logger,
    )

    print(f"\n🚀 Initializing Agent with model '{settings.gemini_model}'...")
    print(f"📦 Workspace directory: {settings.workspace_dir}")
    print(f"🛠️  Enabled tools ({len(registry.list_tools())}): {[t.name for t in registry.list_tools()]}")
    if settings.approve_risky:
        print("🔒 Human-in-the-loop: Risky operations require confirmation.")

    result = agent.run(goal=args.goal)

    if not args.verbose:
        # If not in verbose mode, print a clean summary
        print("\n" + "=" * 60)
        print(f"🏁 Goal Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Iterations: {result.iterations} | Duration: {result.total_duration_s:.2f}s")
        print("=" * 60)
        print("Final Answer:\n")
        print(result.final_answer)
        print("=" * 60)

    print(f"\n📁 Execution log saved to: {logger.log_file}")
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
