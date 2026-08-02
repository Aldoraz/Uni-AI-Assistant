import streamlit as st
from assistant.assistant import Assistant
from context.history import HistoryManager
from llm.openai import OpenAIProvider

def main():
    llm = OpenAIProvider()
    history = HistoryManager()
    assistant = Assistant(
        llm=llm,
        history=history
    )

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