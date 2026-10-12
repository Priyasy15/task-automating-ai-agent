"""Date and time tool providing current time and temporal context."""

from datetime import datetime, timezone
from typing import Any, Type
from pydantic import BaseModel, Field

from agent.tools.base import Tool


class DateTimeArgs(BaseModel):
    timezone_format: str = Field(
        default="local",
        description="Whether to return 'local' time or 'utc' time. Defaults to 'local'.",
    )


class DateTimeTool(Tool):
    """Tool for fetching the current date, time, and timezone information."""

    name: str = "current_datetime"
    description: str = (
        "Returns the current date, local time, UTC time, and day of the week. "
        "Useful for anchoring time-sensitive queries, relative dates, or timestamps."
    )
    args_schema: Type[BaseModel] = DateTimeArgs
    is_risky: bool = False

    def execute(self, timezone_format: str = "local", **kwargs: Any) -> str:
        now_local = datetime.now()
        now_utc = datetime.now(timezone.utc)

        local_str = now_local.strftime("%Y-%m-%d %H:%M:%S")
        utc_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        weekday = now_local.strftime("%A")

        return (
            f"Current Temporal Context:\n"
            f"- Local Time: {local_str} ({weekday})\n"
            f"- UTC Time: {utc_str}\n"
            f"- ISO 8601: {now_local.astimezone().isoformat()}"
        )
