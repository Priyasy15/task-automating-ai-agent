"""Unit tests for ToolRegistry and allowlist mechanisms."""

from agent.tools.calculator import CalculatorTool
from agent.tools.file_ops import FileReadTool, FileWriteTool
from agent.tools.registry import ToolRegistry


def test_registry_register_and_get():
    registry = ToolRegistry()
    calc = CalculatorTool()
    assert registry.register(calc) is True

    retrieved = registry.get_tool("calculator")
    assert retrieved is not None
    assert retrieved.name == "calculator"
    assert registry.get_tool("nonexistent") is None


def test_registry_allowlist_filtering():
    # Only allow calculator and file_read
    registry = ToolRegistry(allowed_tools=["calculator", "file_read"])

    calc = CalculatorTool()
    f_read = FileReadTool()
    f_write = FileWriteTool()

    assert registry.register(calc) is True
    assert registry.register(f_read) is True
    # file_write is not in allowlist
    assert registry.register(f_write) is False

    registered_names = [t.name for t in registry.list_tools()]
    assert "calculator" in registered_names
    assert "file_read" in registered_names
    assert "file_write" not in registered_names


def test_registry_schema_generation():
    registry = ToolRegistry(allowed_tools=["calculator"])
    registry.register(CalculatorTool())

    schemas = registry.get_schemas()
    assert len(schemas) == 1
    calc_schema = schemas[0]

    assert calc_schema["name"] == "calculator"
    assert "parameters_json_schema" in calc_schema
    params = calc_schema["parameters_json_schema"]
    assert params["type"] == "object"
    assert "expression" in params["properties"]
    assert "required" in params and "expression" in params["required"]
