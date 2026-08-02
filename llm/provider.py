from collections.abc import Iterator
from context.entities import Message
from abc import ABC, abstractmethod

class LLMProvider(ABC):

    @abstractmethod
    def chat(self, messages: list[Message]) -> str:
        ...

    @abstractmethod
    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        ...
        