from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage

from app.config import Settings
from app.models.assistant import AssistantResponse, Reply
from app.services.cache import CachingService
from app.services.langchain_llm import LangchainLLMClient


class FakeLLM:
    def __init__(self, responses: list[str], *, raise_error: bool = False) -> None:
        self.responses = responses
        self.raise_error = raise_error
        self.calls: list[dict] = []
        self.bound_kwargs: dict | None = None

    def bind(self, **kwargs):
        self.bound_kwargs = kwargs
        return self

    def with_structured_output(self, *args, **kwargs):
        # In tests we don't need real structured parsing, just return self.
        return self

    async def __call__(self, prompt_input):
        return await self.ainvoke(prompt_input)

    async def ainvoke(self, prompt_input):
        self.calls.append(prompt_input)
        if self.raise_error:
            raise RuntimeError("LLM error")
        payload = self.responses.pop(0) if self.responses else "{}"
        return AIMessage(content=payload, response_metadata={"token_usage": {"total_tokens": 42}})


def _make_settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_model="gpt-test",
        http_timeout_seconds=5,
    )


def test_parse_intent_success():
    response = AssistantResponse(reply=Reply(text="ok")).model_dump_json()
    client = LangchainLLMClient(_make_settings(), llm=FakeLLM([response]), cache=CachingService())

    result = asyncio.run(
        client.parse_intent(
            message="привет",
            profile=None,
            dialog_state=None,
            ui_state=None,
            available_intents=["SHOW_CART"],
        )
    )

    assert result.response.reply.text == "ok"
    assert not result.cached


def test_parse_intent_fallback_on_error():
    client = LangchainLLMClient(_make_settings(), llm=FakeLLM([], raise_error=True), cache=CachingService())

    result = asyncio.run(
        client.parse_intent(
            message="ошибка",
            profile=None,
            dialog_state=None,
            ui_state=None,
            available_intents=[],
        )
    )

    assert "Не удалось" in result.response.reply.text
    assert result.response.actions == []


def test_beautify_reply_parses_json():
    reply_payload = json.dumps({"text": "Красивый ответ", "tone": "friendly"})
    client = LangchainLLMClient(_make_settings(), llm=FakeLLM([reply_payload]), cache=CachingService())
    base_reply = Reply(text="base")

    result = asyncio.run(
        client.beautify_reply(base_reply=base_reply, data={"foo": "bar"}, constraints=None)
    )

    assert result.reply.text == "Красивый ответ"
    assert result.reply.tone == "friendly"
    assert result.cached is False
