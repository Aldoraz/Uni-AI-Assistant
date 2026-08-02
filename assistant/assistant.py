from context.history import HistoryManager
from llm.provider import LLMProvider
from context.entities import Message
from collections.abc import Iterator
from langchain_core.documents import Document
from rag.retriever import Retriever
from pathlib import Path
from context.prompts import SYSTEM_PROMPT_MAIN


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
        self.history.add_message(
            Message(role="user", content=prompt)
        )
        
        documents = self.retriever.retrieve(prompt)
        messages = [
            Message(role="system", content=SYSTEM_PROMPT_MAIN),
            Message(role="system", content=self._format_context(documents)),
            *self.history.get_messages(),
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
        
    def _format_context(self, documents: list[tuple[Document, float]]) -> str:
        if not documents:
            return ""
        
        context = [
        "Use the retrieved context whenever it is relevant.",
        "If the retrieved context does not answer the user's question, ignore it completely and answer from your own knowledge.",
        "Do not invent citations.",
        "Only cite retrieved documents that were actually used.",
        "Similarity scores indicate retrieval quality.",
        "Lower scores indicate more relevant matches.",
        "Treat results below 0.7 as highly relevant.",
        "Treat results above 1.0 as weakly relevant.",
        ""
        ]

        for i, (document, score) in enumerate(documents, start=1):
            context.append(f"--- Document {i} ---")
            context.append(f"Source: {document.metadata.get('filename', 'Unknown')}")
            context.append(f"Page: {document.metadata.get('page', '?') + 1}")
            context.append(f"Similarity Score: {score:.4f}")
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