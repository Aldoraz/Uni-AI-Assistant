from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from context.entities import Message, ToolCall
from tools.tool import ToolDefinition


@dataclass(frozen=True)
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]


class LLMProvider(ABC):

    @abstractmethod
    def chat(self, messages: list[Message]) -> str:
        ...

    @abstractmethod
    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        ...

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> LLMResponse:
        ...

    @abstractmethod
    def stream_chat_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> Iterator[LLMResponse]:
        ...
