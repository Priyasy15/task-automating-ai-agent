"""Web search tool using DuckDuckGo Search (DDGS)."""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS  # Fallback

from agent.tools.base import Tool


class WebSearchArgs(BaseModel):
    query: str = Field(
        ...,
        description="The search query terms to find up-to-date information on the web.",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of search results to retrieve (between 1 and 10).",
    )


class WebSearchTool(Tool):
    """Tool for conducting free web searches via DuckDuckGo."""

    name: str = "web_search"
    description: str = (
        "Searches the web using DuckDuckGo to obtain current facts, news, software "
        "releases, documentation, or general information. Returns titles, URLs, and text snippets."
    )
    args_schema: Type[BaseModel] = WebSearchArgs
    is_risky: bool = False

    def execute(self, query: str, max_results: int = 5, **kwargs: Any) -> str:
        """Execute web search and format results into a structured string."""
        clean_query = query.strip()
        if not clean_query:
            return "Error: Search query cannot be empty."

        try:
            ddgs_client = DDGS()
            raw_results = list(ddgs_client.text(clean_query, max_results=max_results))
            if not raw_results:
                return f"No results found for query: '{clean_query}'."

            formatted_results: List[str] = []
            for i, item in enumerate(raw_results, 1):
                title = item.get("title", "No Title")
                href = item.get("href", item.get("link", "No URL"))
                body = item.get("body", item.get("snippet", "No description available."))
                formatted_results.append(
                    f"[{i}] {title}\nURL: {href}\nSummary: {body}\n"
                )

            return f"Found {len(formatted_results)} results for '{clean_query}':\n\n" + "\n".join(
                formatted_results
            )

        except Exception as e:
            return f"Web Search Error for query '{clean_query}': {type(e).__name__}: {str(e)}"
