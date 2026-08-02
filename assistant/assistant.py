from context.history import HistoryManager
from llm.provider import LLMProvider
from context.entities import Message
from collections.abc import Iterator


class Assistant:
    def __init__(
        self,
        llm: LLMProvider,
        history: HistoryManager,
    ):
        self.llm = llm
        self.history = history
        
    # For internal use like summarization
    def chat(self, messages: list[Message]) -> str:
        return self.llm.chat(messages)


    # For streaming responses to the UI
    def stream_chat(self, prompt: str) -> Iterator[str]:
        self.history.add_message(
            Message(role="user", content=prompt)
        )
        
        stream = self.llm.stream_chat(self.history.get_messages())
        completion = ""
        
        for chunk in stream:
            completion += chunk
            yield chunk  # Pass the chunk to the caller for streaming

        self.history.add_message(
            Message(role="assistant", content=completion)
        )