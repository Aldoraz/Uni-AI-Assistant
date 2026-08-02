from context.entities import Message
import streamlit as st

class HistoryManager:
    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
        self.messages: list[Message] = st.session_state.messages

    def add_message(self, msg: Message):
        self.messages.append(msg)
        
    def get_messages(self) -> list[Message]:
        # TODO: Sliding window instead of full history
        return self.messages.copy()
    
    def clear(self):
        self.messages.clear()