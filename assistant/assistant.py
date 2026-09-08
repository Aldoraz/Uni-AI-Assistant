from collections.abc import Iterator

from langchain_core.documents import Document

from config import LLM_CONTEXT_SIZE, RAG_CONTEXT_SIZE
from context.entities import Message
from context.history import HistoryManager
from context.prompts import SYSTEM_PROMPT_MAIN
from llm.provider import LLMProvider
from rag.retriever import Retriever


class Assistant:
    def __init__(self, llm: LLMProvider, history: HistoryManager, retriever: Retriever):
        self.llm = llm
        self.history = history
        self.retriever = retriever
        
    # For internal use like summarization
    def chat(self, messages: list[Message]) -> str:
        return self.llm.chat(messages)

    # For streaming responses to the UI
    def stream_chat(self, prompt: str) -> Iterator[str]:
        history = self.history.get_messages(limit=RAG_CONTEXT_SIZE)
        documents = self.retriever.retrieve(prompt, history)

        self.history.add_message(
            Message(role="user", content=prompt)
        )

        messages = [
            Message(role="system", content=SYSTEM_PROMPT_MAIN),
            Message(role="system", content=self._format_context(documents)),
            *self.history.get_messages(limit=LLM_CONTEXT_SIZE),
        ]
        self._dump_messages(messages)
        
        
        stream = self.llm.stream_chat(messages)
        completion = ""
        for chunk in stream:
            completion += chunk
            yield chunk  # Pass the chunk to the caller for streaming

        self.history.add_message(
            Message(role="assistant", content=completion)
        )
        
    def _format_context(self, documents: list[Document]) -> str:
        if not documents:
            return ""
        
        context = [
            "Use the retrieved context whenever it is relevant.",
            "If the retrieved context does not answer the user's question, "
            "ignore it completely and answer from your own knowledge.",
            "Do not invent citations.",
            "Only cite retrieved documents that were actually used.",
            "",
        ]

        for i, document in enumerate(documents, start=1):
            context.append(f"--- Document {i} ---")
            context.append(f"Source: {document.metadata.get('filename', 'Unknown')}")
            context.append(f"Page: {document.metadata.get('page', '?') + 1}")
            context.append("")
            context.append(document.page_content.strip())
            context.append("")

        return "\n".join(context)
        
    def _dump_messages(self, messages: list[Message]) -> None:
        with open("prompt_debug.txt", "w", encoding="utf-8") as f:
            for message in messages:
                f.write(f"=== {message.role.upper()} ===\n")
                f.write(message.content)
                f.write("\n\n")
