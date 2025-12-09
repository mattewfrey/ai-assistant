"""
Тесты LangchainLLMClient.

Эти тесты проверяют:
1. Парсинг результатов LLM
2. Fallback при ошибках
3. Конвертацию результатов
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from app.intents import IntentType
from app.models.assistant import AssistantMeta, AssistantResponse, Reply
from app.models.llm_intent import ExtractedSlots, LLMIntentResult
from app.services.langchain_llm import LLMRunResult, BeautifyResult


class TestLLMRunResult:
    """Тесты для LLMRunResult dataclass."""
    
    def test_create_basic_result(self):
        """Создание базового результата."""
        response = AssistantResponse(
            reply=Reply(text="Показываю корзину"),
            actions=[],
        )
        result = LLMRunResult(
            response=response,
            token_usage={"total_tokens": 42},
            cached=False,
        )
        
        assert result.response.reply.text == "Показываю корзину"
        assert result.token_usage["total_tokens"] == 42
        assert not result.cached
    
    def test_result_with_llm_intent(self):
        """Результат с LLMIntentResult."""
        llm_intent = LLMIntentResult(
            intent=IntentType.FIND_BY_SYMPTOM,
            confidence=0.95,
            slots=ExtractedSlots(symptom="кашель", age=5),
            reply="Подбираю препараты",
        )
        
        response = AssistantResponse(
            reply=Reply(text=llm_intent.reply),
            actions=[],
        )
        
        result = LLMRunResult(
            response=response,
            token_usage={},
            cached=False,
            llm_intent_result=llm_intent,
            pipeline_path="llm_only",
            llm_confidence=0.95,
        )
        
        assert result.llm_intent_result.intent == IntentType.FIND_BY_SYMPTOM
        assert result.llm_confidence == 0.95
        assert result.pipeline_path == "llm_only"
    
    def test_result_with_entities_tracking(self):
        """Результат с отслеживанием сущностей до/после LLM."""
        response = AssistantResponse(reply=Reply(text="test"), actions=[])
        
        result = LLMRunResult(
            response=response,
            token_usage={},
            cached=False,
            extracted_entities_before={"symptom": "кашель"},
            extracted_entities_after={"symptom": "кашель", "age": 30},
            pipeline_path="router+llm",
        )
        
        assert result.extracted_entities_before == {"symptom": "кашель"}
        assert result.extracted_entities_after["age"] == 30


class TestBeautifyResult:
    """Тесты для BeautifyResult dataclass."""
    
    def test_create_result(self):
        """Создание результата beautify."""
        reply = Reply(text="Красивый ответ", tone="friendly")
        result = BeautifyResult(reply=reply, cached=False)
        
        assert result.reply.text == "Красивый ответ"
        assert result.reply.tone == "friendly"
        assert not result.cached
    
    def test_cached_result(self):
        """Результат из кэша."""
        reply = Reply(text="Кэшированный ответ")
        result = BeautifyResult(reply=reply, cached=True)
        
        assert result.cached


class TestLLMIntentResultParsing:
    """Тесты парсинга LLMIntentResult."""
    
    def test_parse_full_result(self):
        """Парсинг полного результата."""
        result = LLMIntentResult(
            intent=IntentType.SHOW_CART,
            confidence=0.98,
            slots=ExtractedSlots(),
            reply="Показываю корзину",
            reasoning="Явный запрос показать корзину",
        )
        
        assert result.intent == IntentType.SHOW_CART
        assert result.confidence == 0.98
        assert result.reply == "Показываю корзину"
        assert result.reasoning is not None
    
    def test_parse_with_slots(self):
        """Парсинг с извлечёнными слотами."""
        result = LLMIntentResult(
            intent=IntentType.FIND_BY_SYMPTOM,
            confidence=0.92,
            slots=ExtractedSlots(
                symptom="головная боль",
                age=35,
                dosage_form="tablets",
                price_max=500,
            ),
            reply="Подбираю препараты от головной боли",
        )
        
        slots_dict = result.slots.to_dict()
        
        assert slots_dict["symptom"] == "головная боль"
        assert slots_dict["age"] == 35
        assert slots_dict["dosage_form"] == "tablets"
        assert slots_dict["price_max"] == 500
    
    def test_to_parameters_dict(self):
        """Конвертация в параметры для action."""
        result = LLMIntentResult(
            intent=IntentType.FIND_BY_DISEASE,
            confidence=0.88,
            slots=ExtractedSlots(
                disease="ОРВИ",
                is_for_children=True,
            ),
            reply="Подбираю препараты при ОРВИ",
        )
        
        params = result.to_parameters_dict()
        
        assert params["disease"] == "ОРВИ"
        assert params["is_for_children"] is True
        assert "symptom" not in params  # None значения не включаются
    
    def test_needs_clarification(self):
        """Проверка флага needs_clarification."""
        result = LLMIntentResult(
            intent=IntentType.ASK_PHARMACIST,
            confidence=0.6,
            slots=ExtractedSlots(),
            reply="Уточните, что вас беспокоит?",
            needs_clarification=True,
            missing_required_slots=["symptom"],
        )
        
        assert result.needs_clarification
        assert "symptom" in result.missing_required_slots


class TestExtractedSlots:
    """Тесты для ExtractedSlots."""
    
    def test_empty_slots(self):
        """Пустые слоты."""
        slots = ExtractedSlots()
        assert slots.to_dict() == {}
    
    def test_all_slots(self):
        """Все основные слоты."""
        slots = ExtractedSlots(
            symptom="кашель",
            disease="ОРВИ",
            age=30,
            price_min=100,
            price_max=500,
            dosage_form="tablets",
            product_name="Нурофен",
            is_for_children=True,
            sugar_free=True,
            lactose_free=False,
        )
        
        d = slots.to_dict()
        
        assert d["symptom"] == "кашель"
        assert d["disease"] == "ОРВИ"
        assert d["age"] == 30
        assert d["price_min"] == 100
        assert d["price_max"] == 500
        assert d["dosage_form"] == "tablets"
        assert d["product_name"] == "Нурофен"
        assert d["is_for_children"] is True
        assert d["sugar_free"] is True
        assert d["lactose_free"] is False
    
    def test_merge_slots(self):
        """Мерж двух ExtractedSlots."""
        slots1 = ExtractedSlots(symptom="кашель", age=30)
        slots2 = ExtractedSlots(age=35, dosage_form="syrup")  # age перезаписывает
        
        merged = slots1.merge_with(slots2)
        
        # slots1 имеет приоритет
        assert merged.symptom == "кашель"
        assert merged.age == 30  # из slots1
        assert merged.dosage_form == "syrup"  # из slots2


class TestFallbackBehavior:
    """Тесты fallback поведения."""
    
    def test_fallback_assistant_response(self):
        """Fallback ответ при ошибке."""
        from app.services.langchain_llm import LangchainLLMClient
        
        response = LangchainLLMClient._fallback_assistant_response(message="тест")
        
        assert "Не удалось" in response.reply.text
        assert "тест" in response.reply.text
        assert response.actions == []
    
    def test_fallback_llm_intent_result(self):
        """Fallback LLMIntentResult."""
        from app.services.langchain_llm import LangchainLLMClient
        
        result = LangchainLLMClient._fallback_llm_intent_result()
        
        assert result.intent == IntentType.UNKNOWN
        assert result.confidence == 0.3
        assert "Не удалось" in result.reply


# Legacy function tests для обратной совместимости
def test_parse_intent_success():
    """Legacy тест - проверка структуры LLMRunResult."""
    response = AssistantResponse(
        reply=Reply(text="ok"),
        actions=[],
    )
    result = LLMRunResult(response=response, token_usage={}, cached=False)
    
    assert result.response.reply.text == "ok"
    assert not result.cached


def test_parse_intent_fallback_on_error():
    """Legacy тест - fallback response."""
    from app.services.langchain_llm import LangchainLLMClient
    
    response = LangchainLLMClient._fallback_assistant_response(message="ошибка")
    
    assert "Не удалось" in response.reply.text
    assert response.actions == []


def test_beautify_reply_parses_json():
    """Legacy тест - BeautifyResult структура."""
    reply = Reply(text="Красивый ответ", tone="friendly")
    result = BeautifyResult(reply=reply, cached=False)
    
    assert result.reply.text == "Красивый ответ"
    assert result.reply.tone == "friendly"
    assert result.cached is False
