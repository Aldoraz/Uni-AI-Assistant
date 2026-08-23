# AI-generated with OpenAI Codex

import tempfile
import unittest
from pathlib import Path

from context.conversations import ConversationManager
from context.entities import Message


class ConversationManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name) / "nested" / "conversations.db"
        )
        self.manager = ConversationManager(self.database_path)

    def tearDown(self) -> None:
        self.manager.conn.close()
        self.temporary_directory.cleanup()

    def test_initialization_creates_database_and_tables(self) -> None:
        self.assertTrue(self.database_path.exists())
        tables = {
            row[0]
            for row in self.manager.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        self.assertIn("conversations", tables)
        self.assertIn("messages", tables)

    def test_conversation_crud(self) -> None:
        conversation_id = self.manager.create_conversation("Original")

        self.assertTrue(self.manager.conversation_exists(conversation_id))
        self.assertEqual(
            self.manager.get_conversations(),
            [(conversation_id, "Original")],
        )

        self.manager.rename_conversation(conversation_id, "Renamed")
        self.assertEqual(
            self.manager.get_conversations(),
            [(conversation_id, "Renamed")],
        )

        self.manager.delete_conversation(conversation_id)
        self.assertFalse(self.manager.conversation_exists(conversation_id))

    def test_messages_persist_across_manager_instances(self) -> None:
        conversation_id = self.manager.create_conversation("Persistent")
        messages = [
            Message(role="user", content="question"),
            Message(role="assistant", content="answer"),
        ]
        for message in messages:
            self.manager.add_message(conversation_id, message)

        second_manager = ConversationManager(self.database_path)
        try:
            self.assertEqual(second_manager.get_messages(conversation_id), messages)
        finally:
            second_manager.conn.close()

    def test_message_limit_returns_latest_messages_in_chronological_order(self) -> None:
        conversation_id = self.manager.create_conversation("Limited")
        for content in ("one", "two", "three"):
            self.manager.add_message(
                conversation_id,
                Message(role="user", content=content),
            )

        messages = self.manager.get_messages(conversation_id, limit=2)

        self.assertEqual(
            messages,
            [
                Message(role="user", content="two"),
                Message(role="user", content="three"),
            ],
        )

    def test_limits_must_be_positive(self) -> None:
        conversation_id = self.manager.create_conversation("Invalid limits")

        with self.assertRaises(ValueError):
            self.manager.get_conversations(limit=0)
        with self.assertRaises(ValueError):
            self.manager.get_messages(conversation_id, limit=-1)

    def test_deleting_conversation_cascades_to_messages(self) -> None:
        conversation_id = self.manager.create_conversation("Delete")
        self.manager.add_message(
            conversation_id,
            Message(role="user", content="deleted message"),
        )

        self.manager.delete_conversation(conversation_id)

        message_count = self.manager.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        self.assertEqual(message_count, 0)

    def test_get_conversations_supports_limit(self) -> None:
        self.manager.create_conversation("First")
        self.manager.create_conversation("Second")

        conversations = self.manager.get_conversations(limit=1)

        self.assertEqual(len(conversations), 1)


if __name__ == "__main__":
    unittest.main()
