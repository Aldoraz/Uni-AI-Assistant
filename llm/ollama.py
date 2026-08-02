from collections.abc import Iterator
from context.entities import Message
from llm.provider import LLMProvider

# TODO: Implement

class OllamaChatProvider(LLMProvider):

    def __init__(self, model):
        ...

    def _to_langchain_messages(self, messages: list[Message]):
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]


    def chat(self, messages: list[Message]) -> str:
        ...
    

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        ...

