from pathlib import Path

import streamlit as st
from assistant.assistant import Assistant
from context.conversations import ConversationManager
from context.history import HistoryManager
from llm.openai import OpenAIChatProvider
from llm.ollama import OllamaChatProvider
from embedding.openai import OpenAIEmbeddingProvider
from embedding.ollama import OllamaEmbeddingProvider
from rag.indexer import Indexer
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from config import CHAT_PROVIDER, CHAT_MODEL, EMBEDDING_PROVIDER, EMBEDDING_MODEL, DOCUMENTS_PATH

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

    return llm, embedder


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


@st.cache_resource
def create_rag_resources():
    llm, embedding_provider = create_providers()
    vector_store = VectorStore(embedding_provider)
    retriever = Retriever(vector_store=vector_store, llm=llm)

    return llm, embedding_provider, vector_store, retriever


def main():
    llm, embedding_provider, vector_store, retriever = create_rag_resources()
    conversations = ConversationManager()
    history = HistoryManager(conversations)
    indexer = Indexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store)
    assistant = Assistant(
        llm=llm,
        history=history,
        retriever=retriever
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
