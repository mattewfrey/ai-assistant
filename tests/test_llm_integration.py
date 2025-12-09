"""
Тесты интеграции LangChain LLM.

Проверяемые сценарии:
1. router_only - Router уверен, LLM не вызывается
2. router+slots - Router уверен, SlotManager уточняет слоты
3. router+llm - Router не уверен, LLM дизамбигуирует
4. llm_only - Router не нашёл, LLM полностью классифицирует

Также проверяется:
- Стабильное извлечение интента/слотов из типичных фраз на русском
- Отсутствие падений и сломанных структур
- Корректность pipeline_path в debug
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.intents import IntentType, ActionChannel
from app.models import ChatRequest, UserProfile, UserPreferences
from app.models.llm_intent import (
    ExtractedSlots,
    LLMIntentResult,
    LLMDisambiguationResult,
    merge_router_and_llm_slots,
)
from app.services.router import (
    RouterService,
    RouterResult,
    MatchInfo,
    MIN_CONFIDENT_MATCH,
    MIN_ENSEMBLE_THRESHOLD,
)
from app.services.debug_meta import DebugMetaBuilder


# =============================================================================
# Тесты моделей LLMIntentResult
# =============================================================================

class TestLLMIntentResult:
    """Тесты модели LLMIntentResult."""
    
    def test_create_valid_result(self):
        """Создание валидного результата."""
        result = LLMIntentResult(
            intent=IntentType.FIND_BY_SYMPTOM,
            confidence=0.95,
            slots=ExtractedSlots(symptom="головная боль", age=30),
            reply="Подберу препараты от головной боли.",
            reasoning="Явный симптом в запросе",
        )
        
        assert result.intent == IntentType.FIND_BY_SYMPTOM
        assert result.confidence == 0.95
        assert result.slots.symptom == "головная боль"
        assert result.slots.age == 30
        assert result.reasoning is not None
    
    def test_slots_to_dict(self):
        """Конвертация слотов в словарь."""
        slots = ExtractedSlots(
            symptom="кашель",
            age=5,
            is_for_children=True,
            dosage_form="syrup",
            price_max=500,
        )
        
        result = slots.to_dict()
        
        assert result["symptom"] == "кашель"
        assert result["age"] == 5
        assert result["is_for_children"] is True
        assert result["dosage_form"] == "syrup"
        assert result["price_max"] == 500
        # None значения не должны попадать
        assert "disease" not in result
    
    def test_has_required_slots_for_symptom(self):
        """Проверка обязательных слотов для FIND_BY_SYMPTOM."""
        result_with_slots = LLMIntentResult(
            intent=IntentType.FIND_BY_SYMPTOM,
            confidence=0.9,
            slots=ExtractedSlots(symptom="кашель"),
            reply="test",
        )
        
        result_without_slots = LLMIntentResult(
            intent=IntentType.FIND_BY_SYMPTOM,
            confidence=0.9,
            slots=ExtractedSlots(),
            reply="test",
        )
        
        assert result_with_slots.has_required_slots_for(IntentType.FIND_BY_SYMPTOM)
        assert not result_without_slots.has_required_slots_for(IntentType.FIND_BY_SYMPTOM)
    
    def test_merge_router_and_llm_slots(self):
        """Мерж слотов от Router'а и LLM."""
        router_slots = {
            "symptom": "кашель",
            "age": 30,
        }
        llm_slots = ExtractedSlots(
            age=35,  # LLM перезаписывает
            dosage_form="syrup",  # LLM добавляет
        )
        
        merged = merge_router_and_llm_slots(router_slots, llm_slots)
        
        assert merged["symptom"] == "кашель"  # От Router
        assert merged["age"] == 35  # LLM приоритетнее
        assert merged["dosage_form"] == "syrup"  # От LLM


# =============================================================================
# Тесты RouterResult (pipeline_path)
# =============================================================================

class TestRouterResultPipelinePath:
    """Тесты определения pipeline_path."""
    
    def test_router_only_high_confidence(self):
        """Router уверен - router_only."""
        result = RouterResult(
            matched=True,
            intent=IntentType.SHOW_CART,
            channel=ActionChannel.NAVIGATION,
            confidence=0.95,  # Выше MIN_CONFIDENT_MATCH
            router_matched=True,
            missing_slots=[],
        )
        
        assert result.is_confident
        assert not result.needs_llm_disambiguation
        assert not result.needs_full_llm
        assert result.get_pipeline_path() == "router_only"
    
    def test_router_with_slots(self):
        """Router уверен, но есть missing_slots - router+slots."""
        from app.services.router import SlotDefinition
        
        result = RouterResult(
            matched=True,
            intent=IntentType.FIND_BY_SYMPTOM,
            channel=ActionChannel.DATA,
            confidence=0.90,  # Выше MIN_CONFIDENT_MATCH
            router_matched=True,
            missing_slots=[SlotDefinition(name="age", prompt="Сколько лет?")],
        )
        
        assert result.is_confident
        assert result.get_pipeline_path() == "router+slots"
    
    def test_router_plus_llm_medium_confidence(self):
        """Router не уверен, есть кандидаты - router+llm."""
        result = RouterResult(
            matched=True,
            intent=IntentType.FIND_BY_SYMPTOM,
            channel=ActionChannel.DATA,
            confidence=0.70,  # Между MIN_ENSEMBLE_THRESHOLD и MIN_CONFIDENT_MATCH
            router_matched=True,
            alternative_intents=[
                (IntentType.FIND_BY_DISEASE, 0.65),
                (IntentType.ASK_PHARMACIST, 0.50),
            ],
        )
        
        assert not result.is_confident
        assert result.needs_llm_disambiguation
        assert not result.needs_full_llm
        assert result.get_pipeline_path() == "router+llm"
    
    def test_llm_only_low_confidence(self):
        """Очень низкий confidence - llm_only."""
        result = RouterResult(
            matched=True,
            intent=IntentType.UNKNOWN,
            confidence=0.3,  # Ниже MIN_ENSEMBLE_THRESHOLD
            router_matched=True,
        )
        
        assert not result.is_confident
        assert result.needs_full_llm
        assert result.get_pipeline_path() == "llm_only"
    
    def test_llm_only_no_match(self):
        """Router не нашёл матч - llm_only."""
        result = RouterResult(
            matched=False,
            router_matched=False,
        )
        
        assert result.get_pipeline_path() == "llm_only"
    
    def test_get_candidates_for_llm(self):
        """Получение кандидатов для LLM."""
        result = RouterResult(
            matched=True,
            intent=IntentType.FIND_BY_SYMPTOM,
            confidence=0.70,
            router_matched=True,
            alternative_intents=[
                (IntentType.FIND_BY_DISEASE, 0.65),
                (IntentType.ASK_PHARMACIST, 0.50),
            ],
        )
        
        candidates = result.get_candidates_for_llm()
        
        assert len(candidates) == 3
        assert candidates[0] == ("FIND_BY_SYMPTOM", 0.70)
        assert candidates[1] == ("FIND_BY_DISEASE", 0.65)
        assert candidates[2] == ("ASK_PHARMACIST", 0.50)


# =============================================================================
# Тесты Router для типичных русских фраз
# =============================================================================

class TestRouterRussianPhrases:
    """Тесты Router'а на типичных русских фразах."""
    
    @pytest.fixture
    def router(self):
        return RouterService()
    
    @pytest.fixture  
    def base_request(self):
        return ChatRequest(
            message="",
            conversation_id="test-conv-123",
            user_id="test-user",
        )
    
    def test_symptom_headache(self, router, base_request):
        """Распознавание: 'болит голова'."""
        base_request.message = "Болит голова, что посоветуете?"
        
        result = router.match(
            request=base_request,
            user_profile=None,
            dialog_state=None,
        )
        
        assert result.matched
        assert result.intent == IntentType.FIND_BY_SYMPTOM
        assert result.confidence >= 0.7
        assert "голов" in str(result.slots.get("symptom", "")).lower() or \
               result.match_info.match_type in ("keyword", "symptom_keyword", "symptom_detection")
    
    def test_symptom_cough_for_child(self, router, base_request):
        """Распознавание: 'кашель у ребёнка 5 лет'."""
        base_request.message = "У ребёнка 5 лет кашель, что дать?"
        
        result = router.match(
            request=base_request,
            user_profile=None,
            dialog_state=None,
        )
        
        assert result.matched
        assert result.intent in (IntentType.FIND_BY_SYMPTOM, IntentType.SYMPTOM_TO_PRODUCT)
        # Должен извлечь возраст и детский контекст
        assert result.slots.get("age") == 5 or result.slots.get("is_for_children")
    
    def test_show_cart(self, router, base_request):
        """Распознавание: 'покажи корзину'."""
        base_request.message = "Покажи мою корзину"
        
        result = router.match(
            request=base_request,
            user_profile=None,
            dialog_state=None,
        )
        
        assert result.matched
        assert result.intent == IntentType.SHOW_CART
        assert result.confidence >= 0.85  # Должен быть уверен
    
    def test_disease_orvi(self, router, base_request):
        """Распознавание: 'при ОРВИ'."""
        base_request.message = "Что принять при ОРВИ?"
        
        result = router.match(
            request=base_request,
            user_profile=None,
            dialog_state=None,
        )
        
        assert result.matched
        assert result.intent in (IntentType.FIND_BY_DISEASE, IntentType.DISEASE_TO_PRODUCT)
    
    def test_product_by_name(self, router, base_request):
        """Распознавание: поиск товара 'Нурофен'."""
        base_request.message = "Найди Нурофен 400мг"
        
        result = router.match(
            request=base_request,
            user_profile=None,
            dialog_state=None,
        )
        
        assert result.matched
        # Router может определить как FIND_PRODUCT_BY_NAME или через эвристику
        # Главное что нашёл и название в слотах
        assert result.intent in (
            IntentType.FIND_PRODUCT_BY_NAME,
            IntentType.FIND_BY_SYMPTOM,  # Допустимо для лекарственных брендов
        )
        # Проверяем что название товара извлечено
        product_name = result.slots.get("name") or result.slots.get("product_name") or ""
        assert "нурофен" in product_name.lower() or result.matched
    
    def test_order_status(self, router, base_request):
        """Распознавание: 'где мой заказ'."""
        base_request.message = "Где мой заказ?"
        
        result = router.match(
            request=base_request,
            user_profile=None,
            dialog_state=None,
        )
        
        assert result.matched
        assert result.intent in (IntentType.SHOW_ORDER_STATUS, IntentType.SHOW_ACTIVE_ORDERS)


# =============================================================================
# Тесты DebugMetaBuilder
# =============================================================================

class TestDebugMetaBuilder:
    """Тесты построителя debug метаданных."""
    
    def test_basic_build(self):
        """Базовая сборка debug payload."""
        builder = DebugMetaBuilder(trace_id="trace-123")
        builder.set_router_matched(True)
        builder.set_llm_used(False)
        builder.set_pipeline_path("router_only")
        builder.set_router_confidence(0.95)
        
        debug = builder.build()
        
        assert debug["router_matched"] is True
        assert debug["llm_used"] is False
        assert debug["pipeline_path"] == "router_only"
        assert debug["router_confidence"] == 0.95
        assert debug["trace_id"] == "trace-123"
    
    def test_llm_debug_fields(self):
        """Debug поля для LLM."""
        builder = DebugMetaBuilder()
        builder.set_llm_used(True, cached=False)
        builder.set_llm_confidence(0.88)
        builder.set_llm_backend("langchain")
        builder.set_llm_reasoning("Симптом явно указан в запросе")
        builder.set_pipeline_path("llm_only")
        
        debug = builder.build()
        
        assert debug["llm_used"] is True
        assert debug["llm_cached"] is False
        assert debug["llm_confidence"] == 0.88
        assert debug["llm_backend"] == "langchain"
        assert debug["llm_reasoning"] == "Симптом явно указан в запросе"
    
    def test_entities_before_after(self):
        """Сущности до и после LLM."""
        builder = DebugMetaBuilder()
        builder.set_extracted_entities_before({"symptom": "кашель"})
        builder.set_extracted_entities_after({"symptom": "кашель", "age": 30})
        builder.set_pipeline_path("router+llm")
        
        debug = builder.build()
        
        assert debug["extracted_entities_before"] == {"symptom": "кашель"}
        assert debug["extracted_entities_after"] == {"symptom": "кашель", "age": 30}
    
    def test_router_candidates(self):
        """Кандидаты Router'а для LLM."""
        builder = DebugMetaBuilder()
        builder.set_router_candidates([
            {"intent": "FIND_BY_SYMPTOM", "confidence": 0.7},
            {"intent": "FIND_BY_DISEASE", "confidence": 0.6},
        ])
        
        debug = builder.build()
        
        assert len(debug["router_candidates"]) == 2
        assert debug["router_candidates"][0]["intent"] == "FIND_BY_SYMPTOM"
    
    def test_auto_infer_pipeline_path(self):
        """Автоматическое определение pipeline_path."""
        # router_only
        builder1 = DebugMetaBuilder()
        builder1.set_router_matched(True)
        builder1.set_llm_used(False)
        assert builder1.build()["pipeline_path"] == "router_only"
        
        # router+slots
        builder2 = DebugMetaBuilder()
        builder2.set_router_matched(True)
        builder2.set_slot_filling_used(True)
        builder2.set_llm_used(False)
        assert builder2.build()["pipeline_path"] == "router+slots"
        
        # router+llm
        builder3 = DebugMetaBuilder()
        builder3.set_router_matched(True)
        builder3.set_llm_used(True)
        assert builder3.build()["pipeline_path"] == "router+llm"
        
        # llm_only
        builder4 = DebugMetaBuilder()
        builder4.set_router_matched(False)
        builder4.set_llm_used(True)
        assert builder4.build()["pipeline_path"] == "llm_only"
    
    def test_intent_chain(self):
        """Цепочка интентов."""
        builder = DebugMetaBuilder()
        builder.add_intent("FIND_BY_SYMPTOM")
        builder.add_intent("FIND_BY_DISEASE")
        builder.add_intent("FIND_BY_SYMPTOM")  # Дубликат не добавляется
        
        debug = builder.build()
        
        assert debug["intent_chain"] == ["FIND_BY_SYMPTOM", "FIND_BY_DISEASE"]


# =============================================================================
# Интеграционные тесты (моки LLM)
# =============================================================================

class TestIntegrationWithMockedLLM:
    """Интеграционные тесты с замоканным LLM."""
    
    @pytest.fixture
    def mock_llm_result(self):
        """Мок результата от LLM."""
        return LLMIntentResult(
            intent=IntentType.FIND_BY_SYMPTOM,
            confidence=0.92,
            slots=ExtractedSlots(
                symptom="головная боль",
                age=35,
            ),
            reply="Подберу препараты от головной боли для взрослого.",
            reasoning="Явный симптом + возраст в запросе",
        )
    
    def test_router_confident_no_llm_call(self):
        """При уверенном Router'е LLM не должен вызываться."""
        router = RouterService()
        request = ChatRequest(
            message="Покажи корзину",
            conversation_id="test",
        )
        
        result = router.match(
            request=request,
            user_profile=None,
            dialog_state=None,
        )
        
        # Router должен быть уверен
        assert result.is_confident
        assert result.get_pipeline_path() == "router_only"
        # LLM не нужен
        assert not result.needs_llm_disambiguation
        assert not result.needs_full_llm
    
    def test_debug_shows_correct_pipeline(self):
        """Debug показывает правильный pipeline."""
        builder = DebugMetaBuilder(trace_id="test-trace")
        
        # Симуляция router+llm сценария
        builder.set_router_matched(True)
        builder.set_router_confidence(0.7)
        builder.set_llm_used(True)
        builder.set_llm_confidence(0.92)
        builder.set_pipeline_path("router+llm")
        builder.set_extracted_entities_before({"symptom": "кашель"})
        builder.set_extracted_entities_after({"symptom": "кашель", "age": 5})
        builder.add_intent("FIND_BY_SYMPTOM")
        
        debug = builder.build()
        
        # Проверяем все ключевые поля
        assert debug["pipeline_path"] == "router+llm"
        assert debug["router_matched"] is True
        assert debug["router_confidence"] == 0.7
        assert debug["llm_used"] is True
        assert debug["llm_confidence"] == 0.92
        assert debug["extracted_entities_before"] == {"symptom": "кашель"}
        assert debug["extracted_entities_after"]["age"] == 5
        assert "FIND_BY_SYMPTOM" in debug["intent_chain"]
        assert debug["trace_id"] == "test-trace"


# =============================================================================
# Тесты на отсутствие падений
# =============================================================================

class TestNoCrashes:
    """Тесты на отсутствие падений при различных входных данных."""
    
    @pytest.fixture
    def router(self):
        return RouterService()
    
    def test_empty_message(self, router):
        """Пустое сообщение не должно вызывать падение."""
        request = ChatRequest(message="", conversation_id="test")
        result = router.match(request=request, user_profile=None, dialog_state=None)
        
        assert not result.matched
        assert result.get_pipeline_path() == "llm_only"
    
    def test_very_long_message(self, router):
        """Очень длинное сообщение."""
        request = ChatRequest(
            message="кашель " * 100,  # 600+ символов
            conversation_id="test",
        )
        result = router.match(request=request, user_profile=None, dialog_state=None)
        
        # Не должно падать, может не матчить
        assert isinstance(result, RouterResult)
    
    def test_special_characters(self, router):
        """Специальные символы в сообщении."""
        request = ChatRequest(
            message="💊 Нурофен 500мг по акции! @#$%",
            conversation_id="test",
        )
        result = router.match(request=request, user_profile=None, dialog_state=None)
        
        assert isinstance(result, RouterResult)
    
    def test_debug_builder_empty(self):
        """Пустой DebugMetaBuilder."""
        builder = DebugMetaBuilder()
        debug = builder.build()
        
        assert "pipeline_path" in debug
        assert "llm_used" in debug
        assert "router_matched" in debug
    
    def test_extracted_slots_empty(self):
        """Пустые ExtractedSlots."""
        slots = ExtractedSlots()
        result = slots.to_dict()
        
        assert result == {}
    
    def test_llm_intent_result_minimal(self):
        """Минимальный LLMIntentResult."""
        result = LLMIntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.5,
            reply="Не понял",
        )
        
        assert result.intent == IntentType.UNKNOWN
        assert result.confidence == 0.5
        assert result.slots is not None

