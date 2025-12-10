"""
Тесты для quick_replies при slot-filling.
"""
import pytest
from app.intents import IntentType
from app.services.response_helpers import slot_quick_replies, get_all_slot_quick_replies


class TestSlotQuickReplies:
    """Тесты генерации вариантов быстрого ответа для слотов."""
    
    def test_age_group_quick_replies(self):
        """Проверяем варианты для возрастной группы."""
        replies = slot_quick_replies("age_group", IntentType.FIND_BY_SYMPTOM)
        
        assert len(replies) == 4  # ребёнок, подросток, взрослый, пожилой
        
        labels = [r["label"] for r in replies]
        assert any("Ребёнок" in l for l in labels)
        assert any("Подросток" in l for l in labels)
        assert any("Взрослый" in l for l in labels)
        assert any("Пожилой" in l for l in labels)
        
        # Проверяем структуру
        for reply in replies:
            assert "label" in reply
            assert "query" in reply
            assert "parameters" in reply
            assert "age_group" in reply["parameters"]
    
    def test_pregnancy_status_quick_replies(self):
        """Проверяем варианты для статуса беременности."""
        replies = slot_quick_replies("pregnancy_status")
        
        assert len(replies) == 3  # беременность, кормление, ни то ни другое
        
        params = [r["parameters"].get("pregnancy_status") for r in replies]
        assert "pregnant" in params
        assert "breastfeeding" in params
        assert "none" in params
    
    def test_chronic_conditions_quick_replies(self):
        """Проверяем варианты для хронических заболеваний."""
        replies = slot_quick_replies("chronic_conditions")
        
        assert len(replies) >= 5  # нет, диабет, гипертония, астма, сердце, другое
        
        # Должен быть вариант "нет хронических"
        no_chronic = [r for r in replies if r["parameters"].get("has_chronic_conditions") is False]
        assert len(no_chronic) == 1
        
        # Должны быть варианты с конкретными заболеваниями
        with_conditions = [r for r in replies if r["parameters"].get("chronic_conditions")]
        assert len(with_conditions) >= 3
    
    def test_allergies_quick_replies(self):
        """Проверяем варианты для аллергий."""
        replies = slot_quick_replies("has_allergies")
        
        assert len(replies) == 2  # нет / есть
        
        params = [r["parameters"].get("has_allergies") for r in replies]
        assert False in params
        assert True in params
    
    def test_symptom_duration_quick_replies(self):
        """Проверяем варианты для длительности симптомов."""
        replies = slot_quick_replies("symptom_duration")
        
        assert len(replies) == 4  # сегодня, пару дней, неделю, давно
        
        params = [r["parameters"].get("symptom_duration") for r in replies]
        assert "today" in params
        assert "few_days" in params
        assert "week" in params
        assert "long" in params
    
    def test_symptom_severity_quick_replies(self):
        """Проверяем варианты для выраженности симптомов."""
        replies = slot_quick_replies("symptom_severity")
        
        assert len(replies) == 3  # слабо, умеренно, сильно
        
        params = [r["parameters"].get("symptom_severity") for r in replies]
        assert "mild" in params
        assert "moderate" in params
        assert "severe" in params
    
    def test_dosage_form_quick_replies(self):
        """Проверяем варианты для формы выпуска."""
        replies = slot_quick_replies("dosage_form")
        
        assert len(replies) >= 4  # таблетки, сироп, спрей, капли, любая
        
        labels = [r["label"] for r in replies]
        assert any("Таблетки" in l for l in labels)
        assert any("Сироп" in l for l in labels)
    
    def test_unknown_slot_returns_empty(self):
        """Для неизвестного слота возвращаем пустой список."""
        replies = slot_quick_replies("unknown_slot_xyz")
        assert replies == []
    
    def test_intent_passed_to_replies(self):
        """Проверяем что intent передаётся в parameters."""
        replies = slot_quick_replies("age_group", IntentType.FIND_BY_DISEASE)
        
        for reply in replies:
            assert reply["intent"] == IntentType.FIND_BY_DISEASE.value
    
    def test_get_all_slot_quick_replies(self):
        """Проверяем функцию получения всех quick_replies."""
        all_replies = get_all_slot_quick_replies()
        
        assert "age_group" in all_replies
        assert "pregnancy_status" in all_replies
        assert "chronic_conditions" in all_replies
        assert "has_allergies" in all_replies
        assert "symptom_duration" in all_replies
        assert "symptom_severity" in all_replies
        assert "dosage_form" in all_replies
        
        # Все значения - непустые списки
        for slot_name, replies in all_replies.items():
            assert isinstance(replies, list)
            assert len(replies) > 0, f"Slot {slot_name} should have quick replies"


class TestSlotManagerQuickReplies:
    """Тесты интеграции quick_replies в SlotManager."""
    
    def test_prompt_response_includes_quick_replies(self):
        """Проверяем что ответ с вопросом содержит quick_replies."""
        from app.services.slot_manager import SlotManager
        from app.services.dialog_state_store import get_dialog_state_store
        from app.services.router import RouterResult, SlotDefinition
        
        dialog_store = get_dialog_state_store()
        manager = SlotManager(dialog_state_store=dialog_store)
        
        router_result = RouterResult(
            matched=True,
            intent=IntentType.FIND_BY_SYMPTOM,
            channel=None,
            slots={"symptom": "кашель"},
            missing_slots=[SlotDefinition(name="age_group", prompt="Для кого препарат?")],
            confidence=0.9,
        )
        
        response = manager.handle_router_result(
            router_result=router_result,
            conversation_id="test-qr-1",
            user_profile=None,
        )
        
        # Проверяем наличие quick_replies
        assert response.meta.quick_replies is not None
        assert len(response.meta.quick_replies) > 0
        
        # Проверяем структуру
        first_reply = response.meta.quick_replies[0]
        assert "label" in first_reply
        assert "query" in first_reply
        assert "parameters" in first_reply
        
        # Должны быть варианты возрастных групп
        params = [r["parameters"].get("age_group") for r in response.meta.quick_replies]
        assert "child" in params or "adult" in params

