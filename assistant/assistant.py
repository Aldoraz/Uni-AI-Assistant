import json
import logging
from collections.abc import Iterator

from langchain_core.documents import Document

from config import LLM_CONTEXT_SIZE, MAX_TOOL_CALLS, RAG_CONTEXT_SIZE
from context.entities import Message, ToolCall
from context.history import HistoryManager
from context.prompts import SYSTEM_PROMPT_MAIN, SYSTEM_PROMPT_TOOLS
from llm.provider import LLMProvider
from rag.retriever import Retriever
from tools.tool import ToolDefinition
from tools.tool_orchestrator import ToolOrchestrator

logger = logging.getLogger(__name__)


class Assistant:
    def __init__(
        self,
        llm: LLMProvider,
        history: HistoryManager,
        retriever: Retriever,
        tool_orchestrator: ToolOrchestrator,
    ) -> None:
        self.llm = llm
        self.history = history
        self.retriever = retriever
        self.tool_orchestrator = tool_orchestrator

    def chat(self, messages: list[Message]) -> str:
        return self.llm.chat(messages)

    def stream_chat(self, prompt: str) -> Iterator[str]:
        # Retrieval must see prior history without duplicating the current prompt.
        history = self.history.get_messages(limit=RAG_CONTEXT_SIZE)
        documents = self.retriever.retrieve(prompt, history)
        logger.info(
            "Assistant turn started (history_messages=%d, rag_documents=%d)",
            len(history),
            len(documents),
        )

        self.history.add_message(Message(role="user", content=prompt))

        messages = [
            Message(role="system", content=SYSTEM_PROMPT_MAIN),
            Message(role="system", content=SYSTEM_PROMPT_TOOLS),
            Message(role="system", content=self._format_context(documents)),
            *self.history.get_messages(limit=LLM_CONTEXT_SIZE),
        ]

        tool_definitions = self.tool_orchestrator.get_definitions()
        tool_uses = 0
        while tool_uses < MAX_TOOL_CALLS:
            completion = ""
            tool_calls: list[ToolCall] = []
            self._dump_messages(messages, tool_definitions)

            for event in self.llm.stream_chat_with_tools(
                messages,
                tool_definitions,
            ):
                if event.content:
                    completion += event.content
                    yield event.content

                tool_calls.extend(event.tool_calls)

            if not tool_calls:
                self.history.add_message(Message(role="assistant", content=completion))
                logger.info(
                    "Assistant turn completed (tools_used=%d)",
                    tool_uses,
                )
                return

            messages.append(
                Message(
                    role="assistant",
                    content=completion,
                    tool_calls=tool_calls,
                )
            )

            for tool_call in tool_calls:
                result = self.tool_orchestrator.execute(tool_call)
                tool_uses += 1

                content = result.content
                if result.is_error:
                    content = f"Tool error: {content}"

                messages.append(
                    Message(
                        role="tool",
                        content=content,
                        tool_call_id=tool_call.id,
                    )
                )

        logger.warning(
            "Tool-call limit reached (limit=%d)",
            MAX_TOOL_CALLS,
        )
        messages.append(
            Message(
                role="system",
                content=(
                    "The tool-call limit has been reached. "
                    "Do not request additional tools. Answer using the information "
                    "already available and clearly acknowledge anything uncertain."
                ),
            )
        )

        completion = ""
        self._dump_messages(messages, [])
        for chunk in self.llm.stream_chat(messages):
            completion += chunk
            yield chunk

        self.history.add_message(Message(role="assistant", content=completion))
        logger.info(
            "Assistant turn completed after tool limit (tools_used=%d)",
            tool_uses,
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
            page = document.metadata.get("page")
            page_label = page + 1 if isinstance(page, int) else "?"

            context.append(f"--- Document {i} ---")
            context.append(f"Source: {document.metadata.get('filename', 'Unknown')}")
            context.append(f"Page: {page_label}")
            context.append("")
            context.append(document.page_content.strip())
            context.append("")

        return "\n".join(context)

    def _dump_messages(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> None:
        with open("prompt_debug.txt", "w", encoding="utf-8") as file:
            if tools:
                file.write("=== AVAILABLE TOOLS ===\n")
                for tool in tools:
                    file.write(f"Name: {tool.name}\n")
                    file.write(f"Description: {tool.description}\n")
                    file.write("Input schema:\n")
                    file.write(
                        json.dumps(
                            tool.input_schema,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        )
                    )
                    file.write("\n\n")

            for message in messages:
                file.write(f"=== {message.role.upper()} ===\n")

                if message.tool_call_id is not None:
                    file.write(f"Tool call ID: {message.tool_call_id}\n")

                for tool_call in message.tool_calls:
                    file.write(f"Tool call: {tool_call.name}\n")
                    file.write(f"Tool call ID: {tool_call.id}\n")
                    arguments = json.dumps(
                        tool_call.arguments,
                        ensure_ascii=False,
                        default=str,
                    )
                    file.write(f"Arguments: {arguments}\n")

                file.write(message.content)
                file.write("\n\n")

        logger.info(
            "Prompt debug dump updated (messages=%d, tools=%d)",
            len(messages),
            len(tools),
        )
