# AI-generated with OpenAI Codex

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from context.entities import Message
from context.history import HistoryManager


class FakeSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


class HistoryManagerTests(unittest.TestCase):
    def setUp(self):
        self.session_state = FakeSessionState()
        self.streamlit = SimpleNamespace(session_state=self.session_state)
        self.streamlit_patch = patch("context.history.st", self.streamlit)
        self.streamlit_patch.start()

    def tearDown(self):
        self.streamlit_patch.stop()

    def test_initializes_and_reuses_session_messages(self):
        first = HistoryManager()
        first.add_message(Message(role="user", content="hello"))

        second = HistoryManager()

        self.assertEqual(second.get_messages(), [Message(role="user", content="hello")])

    def test_get_messages_returns_a_copy(self):
        history = HistoryManager()
        snapshot = history.get_messages()

        snapshot.append(Message(role="user", content="not persisted"))

        self.assertEqual(history.get_messages(), [])

    def test_clear_removes_all_messages(self):
        history = HistoryManager()
        history.add_message(Message(role="assistant", content="answer"))

        history.clear()

        self.assertEqual(history.get_messages(), [])


if __name__ == "__main__":
    unittest.main()
