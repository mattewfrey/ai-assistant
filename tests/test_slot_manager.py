from __future__ import annotations

from app.intents import ActionType, IntentType
from app.models import UserProfile, UserPreferences
from app.models.assistant import AssistantResponse
from app.services.dialog_state_store import get_dialog_state_store
from app.services.router import RouterResult, SlotDefinition
from app.services.slot_manager import SlotManager
from app.services.slot_extraction import AgeGroup


# =============================================================================
# Тесты на age_group slot-filling
# =============================================================================


def test_slot_manager_prompts_for_age_group_when_missing():
    """Тест: при отсутствии age_group ассистент задаёт уточняющий вопрос."""
    manager = SlotManager()
    conversation_id = "conv-age-group-1"
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "кашель", "price_max": 300, "dosage_form": "tablets"},
        missing_slots=[SlotDefinition(name="age_group", prompt="Для кого подбираем препарат?")],
        confidence=0.9,
    )

    response = manager.handle_router_result(
        router_result=router_result,
        conversation_id=conversation_id,
        user_profile=None,
    )

    assert isinstance(response, AssistantResponse)
    # Проверяем что это уточняющий вопрос, а не финальный ответ
    assert not response.actions, "Should ask for age_group, not return actions"
    text_lower = response.reply.text.lower()
    assert any(word in text_lower for word in ["кого", "взрослый", "ребёнок", "ребенок", "подросток", "пожилой"])
    # Проверяем debug flags
    assert response.meta and response.meta.debug
    assert response.meta.debug.get("slot_filling_used") is True
    assert response.meta.debug.get("slot_prompt_pending") is True
    assert "age_group" in response.meta.debug.get("pending_slots", [])


def test_slot_manager_fills_age_group_from_child_marker():
    """Тест: при ответе 'для ребёнка' age_group заполняется как child."""
    manager = SlotManager()
    conversation_id = "conv-age-group-child"
    
    # Сначала имитируем запрос без age_group
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "кашель"},
        missing_slots=[SlotDefinition(name="age_group", prompt="Для кого?")],
        confidence=0.9,
    )
    manager.handle_router_result(router_result=router_result, conversation_id=conversation_id, user_profile=None)

    # Теперь отвечаем "для ребёнка 5 лет"
    result = manager.try_handle_followup(
        request_message="для ребёнка 5 лет",
        conversation_id=conversation_id,
        user_profile=None,
    )

    assert result.handled is True
    assert result.slot_filling_used is True
    assert result.assistant_response is not None
    assert result.assistant_response.actions, "Should proceed to action after age_group filled"
    action = result.assistant_response.actions[0]
    assert action.parameters.get("age_group") == AgeGroup.CHILD
    assert action.parameters.get("age") == 5
    # Проверяем debug flags
    assert result.assistant_response.meta and result.assistant_response.meta.debug
    assert result.assistant_response.meta.debug.get("slot_prompt_pending") is False


def test_slot_manager_fills_age_group_from_adult_marker():
    """Тест: при ответе 'взрослый' age_group заполняется как adult."""
    manager = SlotManager()
    conversation_id = "conv-age-group-adult"
    
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "головная боль"},
        missing_slots=[SlotDefinition(name="age_group", prompt="Для кого?")],
        confidence=0.9,
    )
    manager.handle_router_result(router_result=router_result, conversation_id=conversation_id, user_profile=None)

    result = manager.try_handle_followup(
        request_message="взрослому",
        conversation_id=conversation_id,
        user_profile=None,
    )

    assert result.handled is True
    assert result.assistant_response.actions
    action = result.assistant_response.actions[0]
    assert action.parameters.get("age_group") == AgeGroup.ADULT


def test_slot_manager_fills_age_group_from_elderly_marker():
    """Тест: при ответе 'для пожилого' age_group заполняется как elderly."""
    manager = SlotManager()
    conversation_id = "conv-age-group-elderly"
    
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "боль в суставах"},
        missing_slots=[SlotDefinition(name="age_group", prompt="Для кого?")],
        confidence=0.9,
    )
    manager.handle_router_result(router_result=router_result, conversation_id=conversation_id, user_profile=None)

    result = manager.try_handle_followup(
        request_message="для бабушки",
        conversation_id=conversation_id,
        user_profile=None,
    )

    assert result.handled is True
    assert result.assistant_response.actions
    action = result.assistant_response.actions[0]
    assert action.parameters.get("age_group") == AgeGroup.ELDERLY


def test_slot_manager_fills_age_group_from_numeric_age():
    """Тест: при указании числового возраста age_group вычисляется автоматически."""
    manager = SlotManager()
    conversation_id = "conv-age-group-numeric"
    
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "насморк"},
        missing_slots=[SlotDefinition(name="age_group", prompt="Для кого?")],
        confidence=0.9,
    )
    manager.handle_router_result(router_result=router_result, conversation_id=conversation_id, user_profile=None)

    # Отвечаем только возрастом
    result = manager.try_handle_followup(
        request_message="15 лет",
        conversation_id=conversation_id,
        user_profile=None,
    )

    assert result.handled is True
    assert result.assistant_response.actions
    action = result.assistant_response.actions[0]
    assert action.parameters.get("age") == 15
    assert action.parameters.get("age_group") == AgeGroup.TEENAGER  # 15 лет = подросток


def test_slot_manager_no_prompt_when_age_group_present():
    """Тест: если age_group уже есть, уточняющий вопрос не задаётся."""
    manager = SlotManager()
    conversation_id = "conv-age-group-present"
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "кашель", "age_group": "adult"},  # age_group уже есть
        missing_slots=[],  # Нет missing slots
        confidence=0.9,
    )

    response = manager.handle_router_result(
        router_result=router_result,
        conversation_id=conversation_id,
        user_profile=None,
    )

    assert response.actions, "Should proceed to action when age_group is present"
    assert response.meta and response.meta.debug
    assert response.meta.debug.get("slot_prompt_pending") is False


def test_slot_manager_no_prompt_when_numeric_age_present():
    """Тест: если числовой возраст уже есть, не спрашиваем age_group."""
    manager = SlotManager()
    conversation_id = "conv-age-present"
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "температура", "age": 25},  # Возраст уже есть
        missing_slots=[SlotDefinition(name="age_group", prompt="Для кого?")],  # age_group как missing
        confidence=0.9,
    )

    response = manager.handle_router_result(
        router_result=router_result,
        conversation_id=conversation_id,
        user_profile=None,
    )

    # Если возраст уже есть, age_group можно вычислить, поэтому не спрашиваем
    assert response.actions, "Should proceed to action when numeric age is present"


def test_slot_manager_fills_from_profile_age():
    """Тест: age_group вычисляется из возраста в профиле пользователя."""
    manager = SlotManager()
    conversation_id = "conv-profile-age"
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "головная боль"},
        missing_slots=[SlotDefinition(name="age_group", prompt="Для кого?")],
        confidence=0.9,
    )
    # Профиль с возрастом 35 (взрослый)
    profile = UserProfile(user_id="user-profile-age", preferences=UserPreferences(age=35))

    response = manager.handle_router_result(
        router_result=router_result,
        conversation_id=conversation_id,
        user_profile=profile,
    )

    # Профиль содержит возраст, поэтому age_group вычисляется автоматически
    assert response.actions, "Profile age should allow immediate action"


# =============================================================================
# Существующие тесты (совместимость)
# =============================================================================


def test_slot_manager_followup_fills_age_and_returns_action():
    conversation_id = "conv-slot"
    dialog_store = get_dialog_state_store()
    dialog_store.upsert_state(
        conversation_id,
        current_intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "голова болит", "age_group": "adult"},  # Добавили age_group
        pending_slots=["age"],
        slot_questions={"age": "Уточните возраст"},
    )
    manager = SlotManager(dialog_state_store=dialog_store)
    profile = UserProfile(user_id="u1", preferences=UserPreferences())

    result = manager.try_handle_followup(
        request_message="Мне 30",
        conversation_id=conversation_id,
        user_profile=profile,
        debug_builder=None,
    )

    assert result.handled is True
    assert result.slot_filling_used is True
    response = result.assistant_response
    assert response is not None
    assert response.actions
    action = response.actions[0]
    assert action.type == ActionType.CALL_PLATFORM_API
    assert action.parameters.get("age") == 30
    assert response.meta and response.meta.debug
    assert response.meta.debug.get("slot_filling_used") is True


def test_slot_manager_prompts_for_missing_age_group():
    """Тест: запрашивает age_group когда отсутствует."""
    manager = SlotManager()
    conversation_id = "conv-slot-missing-age-group"
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "головная боль"},
        missing_slots=[SlotDefinition(name="age_group", prompt="Для кого подбираем?")],
        confidence=0.9,
    )

    response = manager.handle_router_result(
        router_result=router_result,
        conversation_id=conversation_id,
        user_profile=None,
    )

    assert isinstance(response, AssistantResponse)
    # Проверяем что задан вопрос про возрастную группу
    text_lower = response.reply.text.lower()
    assert any(word in text_lower for word in ["кого", "взрослый", "ребенок", "ребёнок", "подросток", "пожилой"])


def test_slot_manager_handles_followup():
    """Тест: обработка followup с age_group."""
    manager = SlotManager()
    conversation_id = "conv-slot-2"
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "кашель"},
        missing_slots=[SlotDefinition(name="age_group", prompt="Для кого?")],
        confidence=0.9,
    )

    manager.handle_router_result(router_result=router_result, conversation_id=conversation_id, user_profile=None)

    result = manager.try_handle_followup(
        request_message="для ребёнка 7 лет",
        conversation_id=conversation_id,
        user_profile=None,
    )

    assert result.handled
    assert result.assistant_response
    assert result.assistant_response.actions
    # Проверяем что age_group установлен
    action = result.assistant_response.actions[0]
    assert action.parameters.get("age_group") == AgeGroup.CHILD
    assert action.parameters.get("age") == 7


def test_slot_manager_prefills_age_group_from_profile():
    """Тест: age_group заполняется из профиля пользователя."""
    manager = SlotManager()
    conversation_id = "conv-slot-profile"
    router_result = RouterResult(
        matched=True,
        intent=IntentType.FIND_BY_SYMPTOM,
        channel=None,
        slots={"symptom": "ломота"},
        missing_slots=[SlotDefinition(name="age_group", prompt="Для кого?")],
        confidence=0.9,
    )
    # Профиль с возрастом 42 -> adult
    profile = UserProfile(user_id="user-1", preferences=UserPreferences(age=42))

    response = manager.handle_router_result(
        router_result=router_result,
        conversation_id=conversation_id,
        user_profile=profile,
    )

    assert response.actions, "Profile age should allow immediate action without prompt"
