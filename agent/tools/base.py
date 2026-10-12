"""Base Tool class and schema generation using Pydantic."""

from abc import ABC, abstractmethod
import concurrent.futures
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, ValidationError


class Tool(ABC):
    """Abstract base class for all tools available to the AI agent.

    Subclasses must define:
        name: Unique string name of the tool.
        description: Clear explanation of tool behavior and purpose for the LLM.
        args_schema: A Pydantic BaseModel specifying argument names, types, and descriptions.
        is_risky: True if tool modifies persistent state or performs destructive actions.
    """

    name: str
    description: str
    args_schema: Type[BaseModel]
    is_risky: bool = False

    def get_json_schema(self) -> Dict[str, Any]:
        """Generate an OpenAPI-compatible JSON schema from the Pydantic args_schema.

        Returns a dictionary suitable for LLM function declaration `parameters_json_schema`.
        """
        raw_schema = self.args_schema.model_json_schema()
        # Clean schema for strict API compliance
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": raw_schema.get("properties", {}),
        }
        if "required" in raw_schema:
            schema["required"] = raw_schema["required"]
        if "description" in raw_schema:
            schema["description"] = raw_schema["description"]
        return schema

    def validate_args(self, args: Dict[str, Any]) -> BaseModel:
        """Validate input arguments against args_schema.

        Raises:
            ValidationError: If arguments do not conform to schema.
        """
        return self.args_schema.model_validate(args)

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool logic with validated arguments.

        Must return a string representation of the result.
        """
        pass

    def run_with_timeout(
        self,
        args: Dict[str, Any],
        timeout_seconds: float = 20.0,
        max_output_length: int = 2000,
    ) -> str:
        """Safely execute tool with validation, timeout enforcement, and output truncation.

        Never raises an uncaught exception; returns structured error strings so the LLM
        can inspect what went wrong and self-correct.
        """
        # Step 1: Pydantic validation
        try:
            validated = self.validate_args(args)
            validated_kwargs = validated.model_dump()
        except ValidationError as val_err:
            # Format validation errors clearly for LLM self-correction
            error_details = []
            for err in val_err.errors():
                loc = ".".join(str(p) for p in err.get("loc", []))
                msg = err.get("msg", "invalid")
                error_details.append(f"Field '{loc}': {msg}")
            return (
                f"Validation Error for tool '{self.name}': "
                f"{'; '.join(error_details)}. "
                f"Please correct your arguments and try again."
            )
        except Exception as e:
            return f"Argument Parsing Error for tool '{self.name}': {str(e)}"

        # Step 2: Execution with cross-platform timeout via ThreadPoolExecutor
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self.execute, **validated_kwargs)
            result = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            return (
                f"Timeout Error: Tool '{self.name}' timed out after "
                f"{timeout_seconds} seconds. Try a simpler query or alternative tool."
            )
        except Exception as e:
            executor.shutdown(wait=False, cancel_futures=True)
            return f"Execution Error in tool '{self.name}': {type(e).__name__}: {str(e)}"
        else:
            executor.shutdown(wait=False, cancel_futures=True)

        # Step 3: Result truncation to preserve LLM context window
        result_str = str(result)
        if len(result_str) > max_output_length:
            truncated_len = len(result_str)
            result_str = (
                result_str[:max_output_length]
                + f"\n\n... [Output truncated: {truncated_len - max_output_length} characters omitted to save tokens]"
            )

        return result_str
