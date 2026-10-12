"""Safe mathematical expression calculator tool using AST parsing with DoS protection."""

import ast
import math
import operator
from typing import Any, Callable, Dict, Type
from pydantic import BaseModel, Field
from agent.tools.base import Tool

# Safety limits to protect against computational exhaustion / DoS
MAX_EXPRESSION_LENGTH: int = 200
MAX_OPERAND: float = 1e12
MAX_EXPONENT: float = 100
MAX_RESULT: float = 1e100

# Safe binary operators mapping
_OPERATORS: Dict[Type[ast.AST], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Safe unary operators mapping
_UNARY_OPS: Dict[Type[ast.AST], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Whitelisted mathematical functions
_SAFE_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "pow": math.pow,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "ceil": math.ceil,
    "floor": math.floor,
}

# Whitelisted mathematical constants
_SAFE_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node in a restricted environment with DoS checks."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            if abs(node.value) > MAX_OPERAND:
                raise ValueError(
                    f"Operand magnitude ({node.value}) exceeds safe limit of {MAX_OPERAND}."
                )
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    elif isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        raise ValueError(f"Undefined or unauthorized variable: '{node.id}'")

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

        left = _eval_node(node.left)
        right = _eval_node(node.right)

        # DoS check: Exponent capping
        if op_type is ast.Pow:
            if abs(right) > MAX_EXPONENT:
                raise ValueError(
                    f"Exponent {right} exceeds safe limit of {MAX_EXPONENT}."
                )

        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division by zero is not permitted.")

        result = _OPERATORS[op_type](left, right)

        # Result magnitude capping
        if isinstance(result, (int, float)) and abs(result) > MAX_RESULT:
            raise ValueError("Calculation result exceeds safe magnitude limit.")

        return result

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_node(node.operand)
        return _UNARY_OPS[op_type](operand)

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
            func_name = getattr(node.func, "id", "expression")
            raise ValueError(f"Unauthorized or unknown function: '{func_name}'")

        func_name = node.func.id
        func = _SAFE_FUNCTIONS[func_name]
        args = [_eval_node(arg) for arg in node.args]

        if func_name == "pow" and len(args) >= 2:
            if abs(args[1]) > MAX_EXPONENT:
                raise ValueError(
                    f"Exponent {args[1]} exceeds safe limit of {MAX_EXPONENT}."
                )

        result = func(*args)
        if isinstance(result, (int, float)) and abs(result) > MAX_RESULT:
            raise ValueError("Function result exceeds safe magnitude limit.")
        return result

    else:
        raise ValueError(f"Unsupported syntax expression: {type(node).__name__}")


class CalculatorArgs(BaseModel):
    expression: str = Field(
        ...,
        description=(
            "Mathematical expression to evaluate safely. Examples: '25 * 4 + 10', "
            "'sqrt(144)', 'round(15.75 / 3, 2)'."
        ),
    )


class CalculatorTool(Tool):
    """Tool for evaluating mathematical expressions without shell/eval risks."""

    name: str = "calculator"
    description: str = (
        "Safely evaluates mathematical expressions (arithmetic, exponents, roots, "
        "and standard math functions). Protected against DoS and unsafe eval."
    )
    args_schema: Type[BaseModel] = CalculatorArgs
    is_risky: bool = False

    def execute(self, expression: str, **kwargs: Any) -> str:
        """Parse and evaluate the expression safely with length and magnitude caps."""
        clean_expr = expression.strip()
        if not clean_expr:
            return "Error: Expression cannot be empty."

        if len(clean_expr) > MAX_EXPRESSION_LENGTH:
            return (
                f"Calculation Error: Expression exceeds maximum length of "
                f"{MAX_EXPRESSION_LENGTH} characters."
            )

        try:
            parsed = ast.parse(clean_expr, mode="eval")
            result = _eval_node(parsed.body)
            # Format cleanly
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return f"Result: {result}"
        except (SyntaxError, ValueError, ZeroDivisionError) as err:
            return f"Calculation Error: {str(err)}"
        except Exception as e:
            return f"Calculation Error: Unexpected error: {str(e)}"
