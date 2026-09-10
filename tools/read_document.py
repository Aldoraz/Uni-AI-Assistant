import logging

from config import TOOL_MAX_CONTENT_CHARS
from rag.indexer import Indexer
from tools.tool import Tool, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class ReadDocumentTool(Tool):
    definition: ToolDefinition = ToolDefinition(
        name="read_document",
        description=(
            "Read an indexed document when retrieved excerpts are insufficient "
            "or the user explicitly requests the complete document."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the indexed document to read.",
                }
            },
            "required": ["filename"],
            "additionalProperties": False,
        },
    )

    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        filename = arguments.get("filename")

        if not isinstance(filename, str) or not filename.strip():
            return ToolResult(
                content="'filename' must be a non-empty string.",
                is_error=True,
            )

        filename = filename.strip()
        logger.info("Document read started (filename=%r)", filename)

        try:
            documents = self.indexer.load_indexed_document(filename)
            logger.info(
                "Loaded indexed document for tool use (filename=%s, pages=%d)",
                filename,
                len(documents),
            )
            content = "\n\n".join(
                f"--- Page {page_number} ---\n{document.page_content}"
                for page_number, document in enumerate(documents, start=1)
            )

            content_length = len(content)
            if content_length > TOOL_MAX_CONTENT_CHARS:
                logger.info(
                    "Document tool output truncated "
                    "(filename=%s, characters=%d, limit=%d)",
                    filename,
                    content_length,
                    TOOL_MAX_CONTENT_CHARS,
                )
                content = (
                    content[:TOOL_MAX_CONTENT_CHARS]
                    + "\n\n[Document truncated because it exceeded the "
                    "tool-output limit.]"
                )

            return ToolResult(content=content)

        except ValueError as error:
            logger.warning(
                "Document tool could not load file (filename=%s, error=%s)",
                filename,
                error,
            )
            return ToolResult(
                content=str(error),
                is_error=True,
            )
