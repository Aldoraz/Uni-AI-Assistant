from collections.abc import Iterator

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI

from context.entities import Message
from llm.provider import LLMProvider


class OpenAIProvider(LLMProvider):

    def __init__(self, model: str = "gpt-4.1"):
        load_dotenv()

        self.model = ChatOpenAI(model=model)

    def _to_langchain_messages(self, messages: list[Message]):
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]


    def chat(self, messages: list[Message]) -> str:
        response = self.model.invoke(
            self._to_langchain_messages(messages)
        )
        return str(response.content)
    

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        stream = self.model.stream(
            self._to_langchain_messages(messages)
        )

        for chunk in stream:
            if chunk.content:
                yield str(chunk.content)

