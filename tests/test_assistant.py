import unittest

from assistant.assistant import Assistant
from context.entities import Message


class FakeHistory:
    def __init__(self, messages):
        self.messages = list(messages)

    def add_message(self, message):
        self.messages.append(message)

    def get_messages(self):
        return self.messages.copy()


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, history):
        self.calls.append((query, history))
        return []


class FakeLLM:
    def __init__(self):
        self.stream_messages = None

    def chat(self, messages):
        return ""

    def stream_chat(self, messages):
        self.stream_messages = messages
        yield "answer"


class AssistantTests(unittest.TestCase):
    def test_retrieval_uses_prior_history_and_current_prompt_is_added_once(self):
        prior_history = [Message(role="user", content="Earlier question")]
        history = FakeHistory(prior_history)
        retriever = FakeRetriever()
        llm = FakeLLM()
        assistant = Assistant(llm=llm, history=history, retriever=retriever)
        assistant._dump_messages = lambda messages: None

        completion = "".join(assistant.stream_chat("Follow-up question"))

        self.assertEqual(completion, "answer")
        self.assertEqual(retriever.calls, [("Follow-up question", prior_history)])
        user_messages = [
            message
            for message in history.messages
            if message.role == "user" and message.content == "Follow-up question"
        ]
        self.assertEqual(len(user_messages), 1)
        self.assertIn(user_messages[0], llm.stream_messages)
        self.assertEqual(history.messages[-1], Message(role="assistant", content="answer"))


if __name__ == "__main__":
    unittest.main()
