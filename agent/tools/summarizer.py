"""Text summarization and key points extraction tool."""

import re
from typing import Any, List, Optional, Type
from pydantic import BaseModel, Field

from agent.tools.base import Tool


class SummarizerArgs(BaseModel):
    text: str = Field(
        ...,
        description="The source text to summarize or condense.",
    )
    max_bullet_points: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Maximum number of key takeaway bullet points to extract.",
    )
    focus: Optional[str] = Field(
        default=None,
        description="Optional keyword or topic to prioritize when extracting salient points.",
    )


class SummarizerTool(Tool):
    """Tool for distilling raw text into concise executive bullet points."""

    name: str = "summarizer"
    description: str = (
        "Condenses long texts into key takeaway bullet points and executive highlights. "
        "Useful for processing long web search snippets or large documents before outputting."
    )
    args_schema: Type[BaseModel] = SummarizerArgs
    is_risky: bool = False

    def execute(
        self,
        text: str,
        max_bullet_points: int = 5,
        focus: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        clean_text = text.strip()
        if not clean_text:
            return "Error: Text to summarize cannot be empty."

        # Split into sentences or lines
        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+|\n+", clean_text)
            if len(s.strip()) > 15
        ]

        if not raw_sentences:
            return f"Summary: {clean_text[:200]}"

        # Score sentences based on length, position, and optional focus keyword
        focus_lower = focus.lower() if focus else None
        scored_sentences = []
        for idx, sentence in enumerate(raw_sentences):
            score = 1.0
            # Earlier sentences usually have higher information density
            if idx < 3:
                score += 1.5 - (idx * 0.4)
            # Relevance boost if focus term is present
            if focus_lower and focus_lower in sentence.lower():
                score += 3.0
            scored_sentences.append((score, idx, sentence))

        # Select highest-scoring sentences, sorted by original appearance
        selected = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:max_bullet_points]
        selected_in_order = sorted(selected, key=lambda x: x[1])

        bullets = [f"• {item[2]}" for item in selected_in_order]
        word_count = len(clean_text.split())

        result = (
            f"Summary ({word_count} words condensed into {len(bullets)} points):\n"
            + "\n".join(bullets)
        )
        return result
