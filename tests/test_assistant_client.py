from __future__ import annotations

import asyncio

from app.config import Settings
from app.intents import ActionType, IntentType
from app.models import AssistantAction, AssistantResponse, ChatRequest, DataPayload, Reply
from app.services.assistant_client import AssistantClient
from app.services.conversation_store import ConversationStore
from app.services.user_profile_store import UserProfileStore
from app.services.langchain_llm import LLMRunResult


def _build_assistant(use_langchain: bool = False) -> AssistantClient:
    settings = Settings(openai_api_key="" if not use_langchain else "test-key", use_langchain=use_langchain)
    return AssistantClient(
        settings=settings,
        conversation_store=ConversationStore(),
        user_profile_store=UserProfileStore(),
    )


def test_analyze_message_returns_fallback_without_llm():
    assistant = _build_assistant()
    request = ChatRequest(conversation_id="conv", message="Привет")

    response = asyncio.run(assistant.analyze_message(request, intents=[]))

    text = response.reply.text.lower()
    assert "ассистент" in text
    assert "демо" in text or "режим" in text


def test_explain_data_handles_geo_block_error():
    assistant = _build_assistant()
    reply = Reply(text="ok")
    payload = DataPayload(products=[{"id": "p1"}])

    result = asyncio.run(
        assistant.explain_data(original_reply=reply, data=payload, user_message="что купить?")
    )

    assert result is None


class StubLangchainClient:
    def __init__(self, response: AssistantResponse | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    async def parse_intent(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return LLMRunResult(response=self.response, token_usage={}, cached=False)

    async def beautify_reply(self, **kwargs):
        return Reply(text="beautified")


def test_analyze_message_prefers_langchain():
    stub_response = AssistantResponse(
        reply=Reply(text="langchain"),
        actions=[AssistantAction(type=ActionType.NOOP, parameters={})],
    )
    stub = StubLangchainClient(response=stub_response)

    settings = Settings(openai_api_key="key", openai_model="gpt-test", use_langchain=True)
    assistant = AssistantClient(
        settings=settings,
        conversation_store=ConversationStore(),
        user_profile_store=UserProfileStore(),
        langchain_client=stub,
    )
    request = ChatRequest(conversation_id="conv", message="Привет")

    response = asyncio.run(assistant.analyze_message(request, intents=[IntentType.SHOW_CART.value]))

    assert response.reply.text == "langchain"
    assert stub.calls and stub.calls[0]["message"] == "Привет"


def test_analyze_message_falls_back_when_langchain_fails():
    stub = StubLangchainClient(response=None, error=RuntimeError("boom"))

    settings = Settings(openai_api_key="key", openai_model="gpt-test", use_langchain=True)
    assistant = AssistantClient(
        settings=settings,
        conversation_store=ConversationStore(),
        user_profile_store=UserProfileStore(),
        langchain_client=stub,
    )
    request = ChatRequest(conversation_id="conv", message="Привет")

    response = asyncio.run(assistant.analyze_message(request, intents=[IntentType.SHOW_CART.value]))

    assert "Ассистент" in response.reply.text

