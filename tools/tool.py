from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, object]


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False


class Tool(ABC):
    definition: ToolDefinition

    @abstractmethod
    def execute(self, arguments: dict[str, object]) -> ToolResult:
        ...
