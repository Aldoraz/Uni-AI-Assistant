# AI-generated with OpenAI Codex

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from context.conversations import ConversationManager
from context.entities import Message
from context.history import HistoryManager


class FakeSessionState(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value


class HistoryManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "history.db"
        self.conversations = ConversationManager(database_path)
        self.session_state = FakeSessionState()
        self.streamlit = SimpleNamespace(session_state=self.session_state)
        self.streamlit_patch = patch("context.history.st", self.streamlit)
        self.streamlit_patch.start()

    def tearDown(self) -> None:
        self.streamlit_patch.stop()
        self.conversations.conn.close()
        self.temporary_directory.cleanup()

    def test_initialization_creates_and_reuses_active_conversation(self) -> None:
        first = HistoryManager(self.conversations)
        active_id = self.session_state.active_conversation_id
        first.add_message(Message(role="user", content="persisted"))

        second = HistoryManager(self.conversations)

        self.assertEqual(self.session_state.active_conversation_id, active_id)
        self.assertEqual(
            second.get_messages(),
            [Message(role="user", content="persisted")],
        )

    def test_initialization_uses_existing_conversation_without_session_id(self) -> None:
        existing_id = self.conversations.create_conversation("Existing")

        HistoryManager(self.conversations)

        self.assertEqual(self.session_state.active_conversation_id, existing_id)
        self.assertEqual(len(self.conversations.get_conversations()), 1)

    def test_invalid_active_id_falls_back_to_existing_conversation(self) -> None:
        existing_id = self.conversations.create_conversation("Existing")
        self.session_state.active_conversation_id = 999

        HistoryManager(self.conversations)

        self.assertEqual(self.session_state.active_conversation_id, existing_id)

    def test_create_active_conversation_switches_active_id(self) -> None:
        history = HistoryManager(self.conversations)
        original_id = self.session_state.active_conversation_id

        new_id = history.create_active_conversation("Second")

        self.assertNotEqual(new_id, original_id)
        self.assertEqual(self.session_state.active_conversation_id, new_id)

    def test_set_active_conversation_validates_and_switches_id(self) -> None:
        history = HistoryManager(self.conversations)
        second_id = self.conversations.create_conversation("Second")

        history.set_active_conversation(second_id)

        self.assertEqual(self.session_state.active_conversation_id, second_id)
        with self.assertRaises(ValueError):
            history.set_active_conversation(999)

    def test_get_messages_supports_conversation_and_limit(self) -> None:
        history = HistoryManager(self.conversations)
        first_id = self.session_state.active_conversation_id
        history.add_message(Message(role="user", content="one"))
        history.add_message(Message(role="assistant", content="two"))
        second_id = history.create_active_conversation("Second")
        history.add_message(Message(role="user", content="separate"))

        self.assertEqual(
            history.get_messages(first_id, limit=1),
            [Message(role="assistant", content="two")],
        )
        self.assertEqual(
            history.get_messages(second_id),
            [Message(role="user", content="separate")],
        )

    def test_deleting_active_conversation_selects_remaining_conversation(self) -> None:
        history = HistoryManager(self.conversations)
        remaining_id = self.session_state.active_conversation_id
        deleted_id = history.create_active_conversation("Delete me")

        history.delete_conversation(deleted_id)

        self.assertEqual(self.session_state.active_conversation_id, remaining_id)
        self.assertFalse(self.conversations.conversation_exists(deleted_id))

    def test_deleting_only_conversation_creates_replacement(self) -> None:
        history = HistoryManager(self.conversations)
        deleted_id = self.session_state.active_conversation_id

        history.delete_conversation(deleted_id)

        replacement_id = self.session_state.active_conversation_id
        self.assertTrue(self.conversations.conversation_exists(replacement_id))
        self.assertEqual(len(self.conversations.get_conversations()), 1)


if __name__ == "__main__":
    unittest.main()
