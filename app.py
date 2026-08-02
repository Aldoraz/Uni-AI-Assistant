import streamlit as st
from assistant.assistant import Assistant
from context.history import HistoryManager
from llm.openai import OpenAIChatProvider
from llm.ollama import OllamaChatProvider
from embedding.openai import OpenAIEmbeddingProvider
from embedding.ollama import OllamaEmbeddingProvider
from rag.indexer import Indexer
from config import CHAT_PROVIDER, CHAT_MODEL, EMBEDDING_PROVIDER, EMBEDDING_MODEL

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


def main():
    llm, embedding_provider = create_providers()
    history = HistoryManager()
    assistant = Assistant(
        llm=llm,
        history=history
    )
    indexer = Indexer(embedding_provider=embedding_provider)

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