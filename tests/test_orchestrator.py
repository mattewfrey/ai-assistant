from __future__ import annotations

import pytest

from app.config import Settings
from app.intents import ActionChannel, ActionType, IntentType
from app.models import AssistantAction, AssistantResponse, ChatRequest, Reply
from app.services.orchestrator import Orchestrator


class StubAssistantClient:
    """Minimal assistant client stub to skip LLM calls in orchestrator tests."""

    def __init__(self) -> None:
        self._langchain_client = None

    async def beautify_reply(self, *, reply, data, **kwargs):
        return reply


class StubPlatformClient:
    def __init__(self) -> None:
        self.show_cart_calls: list[dict] = []
        self.fetch_products_calls: list[dict] = []

    async def show_cart(self, parameters, request, trace_id=None):
        self.show_cart_calls.append(parameters)
        return {"cart_id": "cart-test", "items": []}

    async def fetch_products(self, intent, parameters, request, user_profile=None, trace_id=None):
        self.fetch_products_calls.append({"intent": intent, "parameters": parameters})
        return [
            {"id": "p1", "price": 350, "intent": getattr(intent, "value", intent)},
            {"id": "p2", "price": 120, "intent": getattr(intent, "value", intent)},
        ]


@pytest.mark.asyncio
async def test_orchestrator_executes_show_cart_action() -> None:
    settings = Settings(openai_api_key="", enable_beautify_reply=False)
    platform = StubPlatformClient()
    assistant = StubAssistantClient()
    orchestrator = Orchestrator(platform_client=platform, assistant_client=assistant, settings=settings)

    assistant_response = AssistantResponse(
        reply=Reply(text="router matched"),
        actions=[
            AssistantAction(
                type=ActionType.CALL_PLATFORM_API,
                channel=ActionChannel.NAVIGATION,
                intent=IntentType.SHOW_CART,
                parameters={},
            )
        ],
    )
    request = ChatRequest(conversation_id="conv-cart", message="Покажи корзину")

    chat_response = await orchestrator.build_response(
        request=request,
        assistant_response=assistant_response,
        router_matched=True,
    )

    assert platform.show_cart_calls, "Ожидали вызов платформы для корзины"
    assert chat_response.data.cart is not None
    assert chat_response.actions[0].intent == IntentType.SHOW_CART


@pytest.mark.asyncio
async def test_orchestrator_fetches_products_for_symptom_intent() -> None:
    settings = Settings(openai_api_key="", enable_beautify_reply=False)
    platform = StubPlatformClient()
    assistant = StubAssistantClient()
    orchestrator = Orchestrator(platform_client=platform, assistant_client=assistant, settings=settings)

    assistant_response = AssistantResponse(
        reply=Reply(text="ищу варианты"),
        actions=[
            AssistantAction(
                type=ActionType.CALL_PLATFORM_API,
                channel=ActionChannel.DATA,
                intent=IntentType.FIND_BY_SYMPTOM,
                parameters={"symptom": "головная боль", "price_max": 500},
            )
        ],
    )
    request = ChatRequest(conversation_id="conv-symptom", message="Болит голова")

    chat_response = await orchestrator.build_response(
        request=request,
        assistant_response=assistant_response,
        router_matched=True,
    )

    assert platform.fetch_products_calls, "Ожидали вызов платформы для списка товаров"
    assert chat_response.data.products, "Должны вернуться товары для симптома"
    assert all(product.get("price", 0) <= 500 for product in chat_response.data.products if product.get("price"))

