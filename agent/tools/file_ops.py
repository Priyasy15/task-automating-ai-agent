"""Sandboxed file operations tools (read, write, list).

All file operations are strictly restricted to the `./workspace` directory.
Path traversal attempts (e.g. '../', absolute paths outside workspace) are blocked.
`file_write` is flagged as a risky action requiring user approval when enabled.
"""

from pathlib import Path
from typing import Any, List, Optional, Type
from pydantic import BaseModel, Field

from agent.tools.base import Tool


def _get_workspace_dir() -> Path:
    """Return the absolute path to the workspace sandbox directory."""
    return Path("./workspace").resolve()


def _resolve_sandboxed_path(file_path: str, workspace_root: Optional[Path] = None) -> Path:
    """Resolve and enforce sandbox boundary for a file path.

    Raises:
        PermissionError: If the resolved path lies outside the sandbox directory.
    """
    workspace_resolved = (workspace_root or _get_workspace_dir()).resolve()
    target = Path(file_path)

    if target.is_absolute():
        resolved_target = target.resolve()
    else:
        resolved_target = (workspace_resolved / target).resolve()

    if not resolved_target.is_relative_to(workspace_resolved):
        raise PermissionError(
            f"Security Violation: Access denied. Path '{file_path}' resolves to '{resolved_target}', "
            f"which escapes the sandboxed workspace directory '{workspace_resolved}'."
        )

    return resolved_target


# =====================================================================
# 1. File Write Tool (Marked as is_risky = True)
# =====================================================================

class FileWriteArgs(BaseModel):
    file_path: str = Field(
        ...,
        description="Relative file path within ./workspace (e.g., 'notes.md' or 'reports/summary.txt').",
    )
    content: str = Field(
        ...,
        description="The text content to write into the file.",
    )
    append: bool = Field(
        default=False,
        description="If True, append to existing content instead of overwriting.",
    )


class FileWriteTool(Tool):
    """Tool for writing content to files inside the sandboxed workspace."""

    name: str = "file_write"
    description: str = (
        "Writes text content to a file inside the sandboxed ./workspace directory. "
        "Can create new files, overwrite existing files, or append text."
    )
    args_schema: Type[BaseModel] = FileWriteArgs
    is_risky: bool = True  # Modifies persistent local state

    def __init__(self, workspace_dir: Optional[Path] = None):
        self.workspace_dir = workspace_dir

    def execute(self, file_path: str, content: str, append: bool = False, **kwargs: Any) -> str:
        try:
            target_path = _resolve_sandboxed_path(file_path, self.workspace_dir)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(target_path, mode, encoding="utf-8") as f:
                f.write(content)

            action = "Appended to" if append else "Wrote"
            char_count = len(content)
            rel_path = target_path.relative_to(self.workspace_dir or _get_workspace_dir())
            return f"Success: {action} {char_count} characters to 'workspace/{rel_path}'."
        except PermissionError as pe:
            return f"Security Error: {str(pe)}"
        except Exception as e:
            return f"File Write Error: {type(e).__name__}: {str(e)}"


# =====================================================================
# 2. File Read Tool
# =====================================================================

class FileReadArgs(BaseModel):
    file_path: str = Field(
        ...,
        description="Relative file path to read from within ./workspace (e.g., 'notes.md').",
    )


class FileReadTool(Tool):
    """Tool for reading text files inside the sandboxed workspace."""

    name: str = "file_read"
    description: str = (
        "Reads the text content of a file located inside the sandboxed ./workspace directory."
    )
    args_schema: Type[BaseModel] = FileReadArgs
    is_risky: bool = False

    def __init__(self, workspace_dir: Optional[Path] = None):
        self.workspace_dir = workspace_dir

    def execute(self, file_path: str, **kwargs: Any) -> str:
        try:
            target_path = _resolve_sandboxed_path(file_path, self.workspace_dir)
            if not target_path.exists():
                return f"File Read Error: File 'workspace/{file_path}' does not exist."
            if target_path.is_dir():
                return f"File Read Error: 'workspace/{file_path}' is a directory, not a file."

            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            rel_path = target_path.relative_to(self.workspace_dir or _get_workspace_dir())
            return f"Contents of 'workspace/{rel_path}':\n\n{content}"
        except PermissionError as pe:
            return f"Security Error: {str(pe)}"
        except Exception as e:
            return f"File Read Error: {type(e).__name__}: {str(e)}"


# =====================================================================
# 3. File List Tool
# =====================================================================

class FileListArgs(BaseModel):
    subpath: str = Field(
        default="",
        description="Optional subdirectory within ./workspace to inspect. Defaults to root workspace.",
    )


class FileListTool(Tool):
    """Tool for listing files and directories inside the sandboxed workspace."""

    name: str = "file_list"
    description: str = (
        "Lists all files and subdirectories located inside the sandboxed ./workspace directory."
    )
    args_schema: Type[BaseModel] = FileListArgs
    is_risky: bool = False

    def __init__(self, workspace_dir: Optional[Path] = None):
        self.workspace_dir = workspace_dir

    def execute(self, subpath: str = "", **kwargs: Any) -> str:
        try:
            target_path = _resolve_sandboxed_path(subpath, self.workspace_dir)
            if not target_path.exists():
                return f"File List Error: Directory '{subpath}' does not exist."
            if not target_path.is_dir():
                return f"File List Error: '{subpath}' is a file, not a directory."

            items: List[str] = []
            for item in sorted(target_path.iterdir()):
                item_type = "DIR " if item.is_dir() else "FILE"
                size_str = f" ({item.stat().st_size} bytes)" if item.is_file() else ""
                items.append(f"[{item_type}] {item.name}{size_str}")

            if not items:
                return f"The workspace directory '{subpath or '.'}' is currently empty."

            return f"Items in workspace directory '{subpath or '.'}':\n" + "\n".join(items)
        except PermissionError as pe:
            return f"Security Error: {str(pe)}"
        except Exception as e:
            return f"File List Error: {type(e).__name__}: {str(e)}"
