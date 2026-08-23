import streamlit as st
from context.conversations import ConversationManager
from context.entities import Message

class HistoryManager:
    def __init__(self, conversations: ConversationManager):
        self.conversations = conversations
        st.session_state.active_conversation_id = self._get_or_create_conversation()


    def add_message(self, msg: Message) -> None:
        self.conversations.add_message(st.session_state.active_conversation_id, msg)


    def get_messages(self, conversation_id: int | None = None, limit: int | None = None) -> list[Message]:
        if conversation_id is None:
            conversation_id = self._get_or_create_conversation()
        return self.conversations.get_messages(conversation_id, limit)


    def create_active_conversation(self, title: str = "New Conversation") -> int:
        new_conv_id = self.conversations.create_conversation(title)
        st.session_state.active_conversation_id = new_conv_id
        return new_conv_id

    def set_active_conversation(self, conversation_id: int) -> None:
        if not self.conversations.conversation_exists(conversation_id):
            raise ValueError(f"Conversation with ID {conversation_id} does not exist.")
        st.session_state.active_conversation_id = conversation_id

    def rename_conversation(self, conversation_id: int, new_title: str) -> None:
        self.conversations.rename_conversation(conversation_id, new_title)
        
    def get_conversations(self, limit: int | None = None) -> list[tuple[int, str]]:
        return self.conversations.get_conversations(limit)


    def delete_conversation(self, conversation_id: int) -> None:
        self.conversations.delete_conversation(conversation_id)
        if st.session_state.active_conversation_id == conversation_id:
            # Set most recent conversation active if it exists, otherwise create a new one
            conversations = self.conversations.get_conversations()
            if conversations:
                st.session_state.active_conversation_id = conversations[0][0]
            else:
                self.create_active_conversation()


    def _get_or_create_conversation(self) -> int:
        active_id = st.session_state.get("active_conversation_id")

        if active_id is not None and self.conversations.conversation_exists(active_id):
            return active_id
        else:
            # Return most recent conversation if it exists, otherwise create a new one
            conversations = self.conversations.get_conversations()
            if conversations:
                return conversations[0][0]
            return self.create_active_conversation()
