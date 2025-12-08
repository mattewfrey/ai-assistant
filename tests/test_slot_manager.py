from __future__ import annotations

from app.intents import IntentType
from app.models import UserPreferences, UserProfile
from app.models.assistant import AssistantResponse
from app.services.dialog_state_store import get_dialog_state_store
from app.services.router import RouterResult, SlotDefinition
from app.services.slot_manager import SlotManager


def test_slot_manager_prompts_for_missing_slots():
    manager = SlotManager()
    conversation_id = "conv-slot"
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "головная боль"},
        missing_slots=[SlotDefinition(name="age", prompt="Укажите возраст")],
        confidence=0.9,
    )

    response = manager.handle_router_result(
        router_result=router_result,
        conversation_id=conversation_id,
        user_profile=None,
    )

    assert isinstance(response, AssistantResponse)
    assert "возраст" in response.reply.text.lower() or "укажите" in response.reply.text.lower()


def test_slot_manager_handles_followup():
    manager = SlotManager()
    conversation_id = "conv-slot-2"
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "кашель"},
        missing_slots=[SlotDefinition(name="age", prompt="Возраст?")],
        confidence=0.9,
    )

    manager.handle_router_result(router_result=router_result, conversation_id=conversation_id, user_profile=None)

    result = manager.try_handle_followup(
        request_message="Нам 7 лет",
        conversation_id=conversation_id,
        user_profile=None,
    )

    assert result.handled
    assert result.assistant_response
    assert result.assistant_response.actions


def test_slot_manager_prefills_age_from_profile():
    manager = SlotManager()
    conversation_id = "conv-slot-profile"
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "ломота"},
        missing_slots=[SlotDefinition(name="age", prompt="Возраст?")],
        confidence=0.9,
    )
    profile = UserProfile(user_id="user-1", preferences=UserPreferences(age=42))

    response = manager.handle_router_result(
        router_result=router_result,
        conversation_id=conversation_id,
        user_profile=profile,
    )

    assert response.actions, "Profile age should allow immediate action without prompt"
