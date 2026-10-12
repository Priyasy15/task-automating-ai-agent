"""Tool registry with allowlist filtering and schema discovery."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from agent.tools.base import Tool
from agent.tools.calculator import CalculatorTool
from agent.tools.datetime_tool import DateTimeTool
from agent.tools.file_ops import FileListTool, FileReadTool, FileWriteTool
from agent.tools.summarizer import SummarizerTool
from agent.tools.web_search import WebSearchTool


class ToolRegistry:
    """Registry managing available agent tools and enforcing security allowlists."""

    def __init__(self, allowed_tools: Optional[List[str]] = None):
        """Initialize the registry with an optional allowlist of tool names.

        Args:
            allowed_tools: If specified, only tools with names in this list can be registered/used.
        """
        self._tools: Dict[str, Tool] = {}
        self._allowed_names: Optional[Set[str]] = set(allowed_tools) if allowed_tools else None

    def register(self, tool: Tool) -> bool:
        """Register a tool instance if it satisfies the allowlist.

        Returns:
            True if registered, False if excluded by allowlist.
        """
        if self._allowed_names is not None and tool.name not in self._allowed_names:
            return False
        self._tools[tool.name] = tool
        return True

    def get_tool(self, name: str) -> Optional[Tool]:
        """Retrieve a registered tool by its name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """Return list of all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return schema declarations for all registered tools."""
        declarations: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            declarations.append({
                "name": tool.name,
                "description": tool.description,
                "parameters_json_schema": tool.get_json_schema(),
            })
        return declarations

    @classmethod
    def create_default(
        cls,
        workspace_dir: Optional[Path] = None,
        allowed_tools: Optional[List[str]] = None,
    ) -> "ToolRegistry":
        """Factory method to instantiate a registry loaded with all standard tools."""
        registry = cls(allowed_tools=allowed_tools)
        default_tools: List[Tool] = [
            CalculatorTool(),
            WebSearchTool(),
            FileReadTool(workspace_dir=workspace_dir),
            FileWriteTool(workspace_dir=workspace_dir),
            FileListTool(workspace_dir=workspace_dir),
            DateTimeTool(),
            SummarizerTool(),
        ]
        for tool in default_tools:
            registry.register(tool)
        return registry
