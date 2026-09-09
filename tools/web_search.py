import logging

from dotenv import load_dotenv
from tavily import TavilyClient

from config import TOOL_MAX_CONTENT_CHARS, WEB_SEARCH_DEPTH, WEB_SEARCH_MAX_RESULTS
from tools.tool import Tool, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    definition: ToolDefinition = ToolDefinition(
        name="web_search",
        description=(
            "Retrieve relevant information from the web using a search engine. "
            "Provide a search query as the input."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to use.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    def __init__(self):
        load_dotenv()
        self.client = TavilyClient()

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        query = arguments.get("query")

        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                content="'query' must be a non-empty string.",
                is_error=True,
            )

        query = query.strip()
        logger.info("Web search started (query=%r)", query)

        try:
            response = self.client.search(
                query=query,
                search_depth=WEB_SEARCH_DEPTH,
                max_results=WEB_SEARCH_MAX_RESULTS,
                include_answer=False,
                include_raw_content=False,
            )
        except Exception as error:
            logger.warning("Web search failed: %s", error)
            return ToolResult(
                content=f"Web search failed: {error}",
                is_error=True,
            )

        results = response.get("results", [])

        if not results:
            logger.info("Web search returned no results")
            return ToolResult(
                content=f"No web results found for '{query}'.",
                is_error=True,
            )

        logger.info(
            "Web search returned results (count=%d): [%s]",
            len(results),
            "; ".join(
                f"{result.get('title', 'Unknown')!r} <{result.get('url', 'Unknown')}>"
                for result in results
            ),
        )

        parts: list[str] = []
        for index, result in enumerate(results, start=1):
            parts.append(
                f"--- Result {index} ---\n"
                f"Title: {result.get('title', 'Unknown')}\n"
                f"URL: {result.get('url', 'Unknown')}\n"
                f"Content: {result.get('content', '')}"
            )
        content = "\n\n".join(parts)

        content_length = len(content)
        if content_length > TOOL_MAX_CONTENT_CHARS:
            logger.info(
                "Web search output truncated (characters=%d, limit=%d)",
                content_length,
                TOOL_MAX_CONTENT_CHARS,
            )
            content = (
                content[:TOOL_MAX_CONTENT_CHARS]
                + "\n\n[Search results truncated because they exceeded the "
                "tool-output limit.]"
            )

        return ToolResult(content=content)
