import logging
from collections.abc import Iterator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama

from context.entities import Message, ToolCall
from llm.provider import LLMProvider, LLMResponse
from tools.tool import ToolDefinition

logger = logging.getLogger(__name__)


class OllamaChatProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        base_url: str,
        context_window: int,
        reasoning: bool,
        keep_alive: str,
    ) -> None:
        self.model = ChatOllama(
            model=model,
            base_url=base_url,
            num_ctx=context_window,
            reasoning=reasoning,
            keep_alive=keep_alive,
            validate_model_on_init=True,
        )
        logger.info("Initialized Ollama chat provider (model=%s)", model)

    def _to_langchain_messages(self, messages: list[Message]) -> list[BaseMessage]:
        converted: list[BaseMessage] = []

        for message in messages:
            if message.role == "system":
                converted.append(SystemMessage(content=message.content))

            elif message.role == "user":
                converted.append(HumanMessage(content=message.content))

            elif message.role == "assistant":
                converted.append(
                    AIMessage(
                        content=message.content,
                        tool_calls=[
                            {
                                "id": call.id,
                                "name": call.name,
                                "args": call.arguments,
                                "type": "tool_call",
                            }
                            for call in message.tool_calls
                        ],
                    )
                )

            elif message.role == "tool":
                if message.tool_call_id is None:
                    raise ValueError("No tool_call_id provided")

                converted.append(
                    ToolMessage(
                        content=message.content,
                        tool_call_id=message.tool_call_id,
                    )
                )

        return converted

    def _to_langchain_tools(
        self,
        tools: list[ToolDefinition],
    ) -> list[dict[str, object]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in tools
        ]

    def chat(self, messages: list[Message]) -> str:
        logger.debug("Sending Ollama chat request (messages=%d)", len(messages))
        response = self.model.invoke(self._to_langchain_messages(messages))
        return str(response.content)

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        logger.debug("Streaming Ollama chat request (messages=%d)", len(messages))
        stream = self.model.stream(self._to_langchain_messages(messages))

        for chunk in stream:
            if chunk.content:
                yield str(chunk.content)

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> LLMResponse:
        logger.debug(
            "Sending Ollama tool-capable request (messages=%d, tools=%d)",
            len(messages),
            len(tools),
        )
        model_with_tools = self.model.bind_tools(self._to_langchain_tools(tools))

        response = model_with_tools.invoke(self._to_langchain_messages(messages))

        tool_calls = [
            ToolCall(
                id=str(call["id"]),
                name=call["name"],
                arguments=call["args"],
            )
            for call in response.tool_calls
        ]

        return LLMResponse(
            content=str(response.content),
            tool_calls=tool_calls,
        )

    def stream_chat_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> Iterator[LLMResponse]:
        logger.debug(
            "Streaming Ollama tool-capable request (messages=%d, tools=%d)",
            len(messages),
            len(tools),
        )
        model_with_tools = self.model.bind_tools(self._to_langchain_tools(tools))

        gathered = None
        for chunk in model_with_tools.stream(self._to_langchain_messages(messages)):
            gathered = chunk if gathered is None else gathered + chunk

            if chunk.content:
                yield LLMResponse(content=str(chunk.content), tool_calls=[])

        if (
            gathered is None
            or not gathered.tool_calls  # pyright: ignore[reportAttributeAccessIssue]
        ):
            return

        yield LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id=str(call["id"]),
                    name=call["name"],
                    arguments=call["args"],
                )
                # The accumulated chunk exposes tool_calls at runtime, but its
                # inferred LangChain union type does not declare that attribute.
                for call in gathered.tool_calls  # pyright: ignore[reportAttributeAccessIssue]
            ],
        )
