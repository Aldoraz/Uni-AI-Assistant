import logging

from context.entities import ToolCall
from tools.tool import Tool, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class ToolOrchestrator:
    def __init__(self, tools: list[Tool]):
        self.tools = {tool.definition.name: tool for tool in tools}

        if len(self.tools) != len(tools):
            raise ValueError("Tool names must be unique")

        logger.info(
            "Registered tools: %s",
            ", ".join(self.tools) if self.tools else "none",
        )

    def get_definitions(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self.tools.values()]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        logger.info(
            "Tool requested (name=%s, call_id=%s)",
            tool_call.name,
            tool_call.id,
        )
        tool = self.tools.get(tool_call.name)

        if not tool:
            logger.warning("Unknown tool requested: %s", tool_call.name)
            return ToolResult(
                content=f"Tool '{tool_call.name}' not found.",
                is_error=True,
            )

        result = tool.execute(tool_call.arguments)
        logger.info(
            "Tool completed (name=%s, call_id=%s, success=%s)",
            tool_call.name,
            tool_call.id,
            not result.is_error,
        )
        return result
