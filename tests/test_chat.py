from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import get_settings
from app.intents import ActionChannel, ActionType, IntentType
from app.main import create_app
from app.models import AssistantAction, AssistantMeta, AssistantResponse, DataPayload, Reply
from app.routers.chat import (
    get_assistant_client,
    get_orchestrator,
    get_router_service,
    get_slot_manager,
    get_user_profile_store_dependency,
)
from app.services.assistant_client import AssistantClient
from app.services.orchestrator import Orchestrator
from app.services.platform_client import PlatformApiClient
from app.services.router import RouterResult
from app.services.slot_manager import SlotHandlingResult
from app.services.user_profile_store import UserProfileStore


class FakeAssistantClient(AssistantClient):
    def __init__(self) -> None:  # pragma: no cover - parent requires settings
        self.calls = 0
        self._langchain_client = None

    async def analyze_message(self, request, intents):  # type: ignore[override]
        self.calls += 1
        return AssistantResponse(
            reply=Reply(text="ok"),
            actions=[
                AssistantAction(
                    type=ActionType.CALL_PLATFORM_API,
                    intent=IntentType.FIND_BY_SYMPTOM,
                    parameters={"symptom": "кашель", "age": 30},
                )
            ],
        )

    async def explain_data(self, original_reply, data, user_message):  # type: ignore[override]
        return None

    async def beautify_reply(self, *, reply, data, constraints=None, user_message=None, **kwargs):  # type: ignore[override]
        return reply


class FakePlatformClient(PlatformApiClient):
    def __init__(self) -> None:  # pragma: no cover
        self._mock = None  # align with base client attribute usage

    async def fetch_products(  # type: ignore[override]
        self,
        intent,
        parameters,
        request,
        *,
        user_profile=None,
        trace_id=None,
        **kwargs,
    ):
        return [{"id": "p1"}]

    async def dispatch(self, action, request, *, user_profile=None, trace_id=None, **kwargs):  # type: ignore[override]
        return DataPayload(products=[{"id": "p1"}])

    async def show_cart(self, parameters, request, trace_id=None, **kwargs):  # type: ignore[override]
        return {"cart": {"items": []}}


def create_test_app(router_service=None, slot_manager=None) -> TestClient:
    app = create_app()
    settings = get_settings()

    fake_assistant = FakeAssistantClient()
    fake_platform = FakePlatformClient()
    profile_store = UserProfileStore()

    def _get_assistant():
        return fake_assistant

    def _get_orchestrator():
        return Orchestrator(
            platform_client=fake_platform,
            assistant_client=fake_assistant,
            settings=settings,
            user_profile_store=profile_store,
        )

    app.dependency_overrides[get_assistant_client] = _get_assistant
    app.dependency_overrides[get_orchestrator] = _get_orchestrator
    app.dependency_overrides[get_user_profile_store_dependency] = lambda: profile_store
    if router_service is not None:
        app.dependency_overrides[get_router_service] = lambda: router_service
    if slot_manager is not None:
        app.dependency_overrides[get_slot_manager] = lambda: slot_manager

    app.state.fake_assistant = fake_assistant

    return TestClient(app)


def test_assistant_response_validation():
    payload = {
        "reply": {"text": "Привет"},
        "actions": [
            {"type": "CALL_PLATFORM_API", "intent": "FIND_BY_SYMPTOM", "parameters": {"symptom": "кашель"}}
        ],
        "meta": {"confidence": 0.9},
    }
    response = AssistantResponse.model_validate(payload)
    assert response.reply.text == "Привет"
    assert response.actions[0].intent == IntentType.FIND_BY_SYMPTOM


def test_post_message_flow():
    client = create_test_app()
    payload = {
        "message": "Подскажи препараты",
        "ui_state": {"entry_point": "chat_screen"},
    }
    resp = client.post("/api/ai/chat/message", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["reply"]["text"]
    assert data["data"]["products"]


def test_local_router_shortcuts_llm():
    class ShortcutRouter:
        def match(self, **kwargs):
            return RouterResult(
                matched=True,
                intent=IntentType.FIND_BY_SYMPTOM,
                channel=ActionChannel.DATA,
                slots={"symptom": "кашель", "age": 30},
                missing_slots=[],
                confidence=1.0,
            )

    class PassthroughSlotManager:
        def try_handle_followup(self, **kwargs):
            return SlotHandlingResult(handled=False)

        def handle_router_result(self, *, router_result, conversation_id, user_profile=None):
            action = AssistantAction(
                type=ActionType.CALL_PLATFORM_API,
                channel=router_result.channel,
                intent=router_result.intent,
                parameters=router_result.slots,
            )
            return AssistantResponse(
                reply=Reply(text="router handled"),
                actions=[action],
                meta=AssistantMeta(
                    confidence=1.0,
                    top_intent=router_result.intent.value,
                    debug={"slot_filling_used": False, "slot_prompt_pending": False},
                ),
            )

    client = create_test_app(router_service=ShortcutRouter(), slot_manager=PassthroughSlotManager())
    payload = {"message": "Покажи корзину"}

    resp = client.post("/api/ai/chat/message", json=payload)
    assert resp.status_code == 200, resp.text
    fake_assistant = client.app.state.fake_assistant
    assert fake_assistant.calls == 0

