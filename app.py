import logging
from pathlib import Path

from colorama import Fore, Style, just_fix_windows_console
import streamlit as st

from assistant.assistant import Assistant
from config import (
    CHAT_MODEL,
    CHAT_PROVIDER,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
)
from context.conversations import ConversationManager
from context.history import HistoryManager
from embedding.ollama import OllamaEmbeddingProvider
from embedding.openai import OpenAIEmbeddingProvider
from llm.ollama import OllamaChatProvider
from llm.openai import OpenAIChatProvider
from rag.indexer import Indexer
from rag.retriever import Retriever
from rag.vector_store import VectorStore
from tools.read_document import ReadDocumentTool
from tools.tool_orchestrator import ToolOrchestrator
from tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)


class ColoredLevelFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def __init__(self, format_string: str, use_color: bool):
        super().__init__(format_string)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        if self.use_color:
            color = self.LEVEL_COLORS.get(record.levelno, "")
            record.levelname = (
                f"{color}{original_levelname:<8}{Style.RESET_ALL}"
            )
        else:
            record.levelname = f"{original_levelname:<8}"

        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def configure_logging() -> None:
    just_fix_windows_console()
    format_string = "%(levelname)s | %(asctime)s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=format_string,
    )

    for handler in logging.getLogger().handlers:
        stream = getattr(handler, "stream", None)
        is_terminal = bool(getattr(stream, "isatty", lambda: False)())
        handler.setFormatter(ColoredLevelFormatter(format_string, is_terminal))

    for application_logger in (
        "assistant",
        "context",
        "embedding",
        "llm",
        "rag",
        "tools",
        __name__,
    ):
        logging.getLogger(application_logger).setLevel(logging.INFO)

    for noisy_logger in ("faiss.loader", "httpcore", "httpx", "openai", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def render_conversation_sidebar(history: HistoryManager) -> None:
    with st.sidebar:
        st.header("Conversations")

        if st.button("New conversation"):
            history.create_active_conversation()
            st.rerun()

        conversations = history.get_conversations()
        conversation_ids = [conversation_id for conversation_id, _ in conversations]
        titles = dict(conversations)
        active_id = int(st.session_state.active_conversation_id)

        def conversation_title(conversation_id: int) -> str:
            return titles[conversation_id]

        selected_id = st.selectbox(
            "Conversation",
            options=conversation_ids,
            index=conversation_ids.index(active_id),
            format_func=conversation_title,
        )

        if selected_id != active_id:
            history.set_active_conversation(selected_id)
            st.rerun()

        with st.form(f"rename_conversation_{active_id}"):
            new_title = st.text_input("Conversation title", value=titles[active_id])
            rename_submitted = st.form_submit_button("Rename")

        if rename_submitted:
            new_title = new_title.strip()
            if new_title:
                history.rename_conversation(active_id, new_title)
                st.rerun()
            else:
                st.error("Conversation title cannot be empty.")

        if st.button("Delete conversation"):
            history.delete_conversation(active_id)
            st.rerun()


def render_index_status(indexer: Indexer) -> None:
    with st.sidebar.expander("Indexed documents"):
        st.header("Documents")
        if st.button("Update Index"):
            with st.spinner("Reindexing documents..."):
                result = indexer.index_folder(DOCUMENTS_PATH)
                st.success(
                    f"Index updated: "
                    f"{result.files_loaded} loaded, "
                    f"{result.files_skipped} skipped, "
                    f"{result.files_failed} failed, "
                    f"{result.vectors_stored} vectors stored."
                )

        records = indexer.get_index_status()
        if not records:
            st.caption("No indexed documents found.")
        else:
            status_icons = {
                "succeeded": "🟢",
                "failed": "🔴",
                "skipped": "🟡",
            }

            for record in records:
                filename = Path(record.path).name
                icon = status_icons.get(record.index_status, "⚪")

                with st.expander(f"{icon} {filename}"):
                    st.write(f"Chunks: {record.chunk_count}")

                    if record.error_message:
                        st.error(record.error_message)


def create_providers():
    # Chat model selection
    if CHAT_PROVIDER == "openai":
        llm = OpenAIChatProvider(model=CHAT_MODEL)
    elif CHAT_PROVIDER == "ollama":
        llm = OllamaChatProvider(model=CHAT_MODEL)
        pass
    else:
        raise ValueError(f"Unsupported chat provider: {CHAT_PROVIDER}")

    # Embedding model selection
    if EMBEDDING_PROVIDER == "openai":
        embedder = OpenAIEmbeddingProvider(model=EMBEDDING_MODEL)
    elif EMBEDDING_PROVIDER == "ollama":
        embedder = OllamaEmbeddingProvider(model=EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")

    logger.info(
        "Providers configured (chat=%s/%s, embedding=%s/%s)",
        CHAT_PROVIDER,
        CHAT_MODEL,
        EMBEDDING_PROVIDER,
        EMBEDDING_MODEL,
    )
    return llm, embedder


@st.cache_resource
def create_rag_resources():
    llm, embedding_provider = create_providers()
    vector_store = VectorStore(embedding_provider)
    retriever = Retriever(vector_store=vector_store, llm=llm)

    return llm, embedding_provider, vector_store, retriever

def create_tool_orchestrator(indexer: Indexer) -> ToolOrchestrator:
    return ToolOrchestrator(
        tools=[
            ReadDocumentTool(indexer),
            WebSearchTool(),
        ]
    )


def main():
    configure_logging()
    llm, embedding_provider, vector_store, retriever = create_rag_resources()
    conversations = ConversationManager()
    history = HistoryManager(conversations)
    indexer = Indexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store)
    tool_orchestrator = create_tool_orchestrator(indexer=indexer)
    assistant = Assistant(
        llm=llm,
        history=history,
        retriever=retriever,
        tool_orchestrator=tool_orchestrator,
    )

    render_conversation_sidebar(history)
    render_index_status(indexer)

    st.title("Learning AI Assistant")
    for message in history.get_messages():
        with st.chat_message(message.role):
            st.markdown(message.content)

    if prompt := st.chat_input():
        # Immediate feedback to the user
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.write_stream(
                assistant.stream_chat(prompt)
            )

if __name__ == "__main__":
    main()
