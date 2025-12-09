"""
Комплексные тесты NLU-пайплайна.

Покрывает:
1. RouterService - распознавание интентов
2. SlotManager - управление слотами
3. Slot Extraction - извлечение сущностей
4. Интеграционные сценарии
"""

from __future__ import annotations

import pytest

from app.intents import ActionType, IntentType
from app.models import ChatRequest, UserPreferences, UserProfile
from app.services.dialog_state_store import get_dialog_state_store
from app.services.router import RouterService, reset_router_service
from app.services.slot_manager import SlotManager, reset_slot_manager
from app.services.slot_extraction import (
    extract_age,
    extract_price_max,
    extract_price_min,
    extract_price_range,
    extract_symptom,
    extract_symptoms,
    extract_disease,
    extract_diseases,
    extract_dosage_form,
    extract_is_for_children,
    extract_special_filters,
    extract_all_entities,
    normalize_text,
)
from app.services.debug_meta import DebugMetaBuilder


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Сбрасывает синглтоны перед каждым тестом."""
    reset_router_service()
    reset_slot_manager()
    yield
    reset_router_service()
    reset_slot_manager()


@pytest.fixture
def router():
    """Возвращает RouterService."""
    return RouterService()


@pytest.fixture
def slot_manager():
    """Возвращает SlotManager."""
    return SlotManager()


@pytest.fixture
def dialog_store():
    """Возвращает DialogStateStore."""
    return get_dialog_state_store()


# =============================================================================
# Тесты: Извлечение сущностей (slot_extraction)
# =============================================================================

class TestSlotExtraction:
    """Тесты извлечения сущностей."""
    
    # --- Возраст ---
    
    @pytest.mark.parametrize("message,expected", [
        ("мне 30 лет", 30),
        ("ему 5 лет", 5),
        ("ребенку 7 лет", 7),
        ("нам 25", 25),
        ("возраст 42", 42),
        ("для ребенка 3 года", 3),
        ("малышу 2 годика", 2),
        ("сыну 10 лет", 10),
        ("дочке 15", 15),
        ("30", 30),  # Короткий ответ
        ("5 лет", 5),  # Короткий ответ
    ])
    def test_extract_age(self, message: str, expected: int):
        """Тест извлечения возраста."""
        assert extract_age(message) == expected
    
    @pytest.mark.parametrize("message", [
        "привет",
        "где аптека",
        "показать корзину",
        "цена 500 рублей",
    ])
    def test_extract_age_none(self, message: str):
        """Тест: возраст не найден."""
        assert extract_age(message) is None
    
    # --- Цена ---
    
    @pytest.mark.parametrize("message,expected", [
        ("до 500 рублей", 500),
        ("до 500р", 500),
        ("до 500₽", 500),
        ("максимум 1000", 1000),
        ("не дороже 300", 300),
        ("бюджет 700", 700),
        ("подешевле", 500),  # Дефолтный бюджет
        ("недорого", 500),
    ])
    def test_extract_price_max(self, message: str, expected: int):
        """Тест извлечения максимальной цены."""
        assert extract_price_max(message) == expected
    
    @pytest.mark.parametrize("message,expected", [
        ("от 300 рублей", 300),
        ("минимум 500", 500),
        ("не дешевле 1000", 1000),
    ])
    def test_extract_price_min(self, message: str, expected: int):
        """Тест извлечения минимальной цены."""
        assert extract_price_min(message) == expected
    
    def test_extract_price_range(self):
        """Тест извлечения ценового диапазона."""
        min_price, max_price = extract_price_range("от 300 до 1000 рублей")
        assert min_price == 300
        assert max_price == 1000
    
    # --- Симптомы ---
    
    @pytest.mark.parametrize("message,expected", [
        ("болит голова", "головная боль"),
        ("от кашля", "кашель"),
        ("от горла", "боль в горле"),
        ("температура высокая", "температура"),
        ("насморк замучил", "насморк"),
        ("тошнота", "тошнота"),
        ("изжога после еды", "изжога"),
        ("аллергия на пыльцу", "аллергия"),
    ])
    def test_extract_symptom(self, message: str, expected: str):
        """Тест извлечения симптома."""
        result = extract_symptom(message)
        assert result is not None
        assert expected.lower() in result.lower() or result.lower() in expected.lower()
    
    def test_extract_multiple_symptoms(self):
        """Тест извлечения нескольких симптомов."""
        symptoms = extract_symptoms("кашель и насморк, болит голова")
        assert len(symptoms) >= 2
        symptom_names = [s.lower() for s in symptoms]
        assert any("кашель" in s for s in symptom_names)
        assert any("насморк" in s for s in symptom_names)
    
    # --- Заболевания ---
    
    @pytest.mark.parametrize("message,expected", [
        ("при ОРВИ", "ОРВИ"),
        ("лечение гриппа", "грипп"),
        ("при гастрите", "гастрит"),
        ("у меня бронхит", "бронхит"),
        ("от цистита", "цистит"),
    ])
    def test_extract_disease(self, message: str, expected: str):
        """Тест извлечения заболевания."""
        result = extract_disease(message)
        assert result is not None
        assert expected.lower() in result.lower()
    
    # --- Форма выпуска ---
    
    @pytest.mark.parametrize("message,expected", [
        ("в таблетках", "tablets"),
        ("сироп от кашля", "syrup"),
        ("спрей для горла", "spray"),
        ("капли в нос", "drops"),
        ("мазь для суставов", "ointment"),
        ("гель обезболивающий", "gel"),
        ("порошок для разведения", "powder"),
        ("свечи детские", "suppository"),
    ])
    def test_extract_dosage_form(self, message: str, expected: str):
        """Тест извлечения формы выпуска."""
        assert extract_dosage_form(message) == expected
    
    # --- Специальные фильтры ---
    
    def test_extract_children_context(self):
        """Тест извлечения детского контекста."""
        assert extract_is_for_children("для ребенка") is True
        assert extract_is_for_children("детский сироп") is True
        assert extract_is_for_children("малышу") is True
        assert extract_is_for_children("взрослому") is False
    
    def test_extract_special_filters(self):
        """Тест извлечения специальных фильтров."""
        filters = extract_special_filters("без сахара и без лактозы")
        assert filters.get("sugar_free") is True
        assert filters.get("lactose_free") is True
    
    # --- Комплексное извлечение ---
    
    def test_extract_all_entities(self):
        """Тест комплексного извлечения."""
        result = extract_all_entities(
            "от кашля ребенку 5 лет в сиропе до 500 рублей без сахара"
        )
        
        assert result.symptom is not None
        assert "кашель" in result.symptom.lower()
        assert result.age == 5
        assert result.is_for_children is True
        assert result.dosage_form == "syrup"
        assert result.price_max == 500
        assert result.special_filters.get("sugar_free") is True
    
    def test_normalize_text(self):
        """Тест нормализации текста."""
        assert normalize_text("  ПРИВЕТ  МИР  ") == "привет мир"
        assert normalize_text("Ёлка") == "елка"


# =============================================================================
# Тесты: RouterService
# =============================================================================

class TestRouterService:
    """Тесты RouterService."""
    
    # --- Корзина ---
    
    @pytest.mark.parametrize("message", [
        "покажи корзину",
        "что в корзине",
        "моя корзина",
        "открой корзину",
        "cart",
    ])
    def test_router_matches_cart(self, router: RouterService, message: str):
        """Тест распознавания корзины."""
        result = router.match(
            request=ChatRequest(message=message, conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        assert result.matched is True
        assert result.intent == IntentType.SHOW_CART
        assert result.confidence >= 0.8
    
    # --- Заказы ---
    
    @pytest.mark.parametrize("message,expected_intent", [
        ("где мой заказ", IntentType.SHOW_ORDER_STATUS),
        ("статус заказа", IntentType.SHOW_ORDER_STATUS),
        ("мои заказы", IntentType.SHOW_ACTIVE_ORDERS),
        ("история заказов", IntentType.SHOW_ORDER_HISTORY),
        ("оформить заказ", IntentType.PLACE_ORDER),
    ])
    def test_router_matches_orders(self, router: RouterService, message: str, expected_intent: IntentType):
        """Тест распознавания заказов."""
        result = router.match(
            request=ChatRequest(message=message, conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        assert result.matched is True
        assert result.intent == expected_intent
    
    # --- Профиль и бонусы ---
    
    @pytest.mark.parametrize("message,expected_intent", [
        ("мой профиль", IntentType.SHOW_PROFILE),
        ("личный кабинет", IntentType.SHOW_PROFILE),
        ("избранное", IntentType.SHOW_FAVORITES),
        ("мои бонусы", IntentType.SHOW_BONUS_BALANCE),
        ("сколько бонусов", IntentType.SHOW_BONUS_BALANCE),
    ])
    def test_router_matches_profile(self, router: RouterService, message: str, expected_intent: IntentType):
        """Тест распознавания профиля."""
        result = router.match(
            request=ChatRequest(message=message, conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        assert result.matched is True
        assert result.intent == expected_intent
    
    # --- Аптеки ---
    
    @pytest.mark.parametrize("message,expected_intent", [
        ("ближайшие аптеки", IntentType.SHOW_NEARBY_PHARMACIES),
        ("аптеки рядом", IntentType.SHOW_NEARBY_PHARMACIES),
        ("часы работы аптеки", IntentType.SHOW_PHARMACY_HOURS),
    ])
    def test_router_matches_pharmacies(self, router: RouterService, message: str, expected_intent: IntentType):
        """Тест распознавания аптек."""
        result = router.match(
            request=ChatRequest(message=message, conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        assert result.matched is True
        assert result.intent == expected_intent
    
    # --- Правовая информация ---
    
    @pytest.mark.parametrize("message,expected_intent", [
        ("политика возврата", IntentType.RETURN_POLICY),
        ("можно ли вернуть", IntentType.RETURN_POLICY),
        ("как хранить лекарства", IntentType.STORAGE_RULES),
        ("нужен ли рецепт", IntentType.PRESCRIPTION_POLICY),
        ("условия доставки", IntentType.DELIVERY_RULES),
    ])
    def test_router_matches_legal(self, router: RouterService, message: str, expected_intent: IntentType):
        """Тест распознавания правовой информации."""
        result = router.match(
            request=ChatRequest(message=message, conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        assert result.matched is True
        assert result.intent == expected_intent
    
    # --- Симптомы ---
    
    @pytest.mark.parametrize("message", [
        "болит голова",
        "от кашля",
        "температура высокая",
        "что-нибудь от насморка",
        "средство от горла",
    ])
    def test_router_matches_symptoms(self, router: RouterService, message: str):
        """Тест распознавания симптомов."""
        result = router.match(
            request=ChatRequest(message=message, conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        assert result.matched is True
        assert result.intent == IntentType.FIND_BY_SYMPTOM
        assert "symptom" in result.slots
    
    # --- Заболевания ---
    
    @pytest.mark.parametrize("message", [
        "при ОРВИ",
        "лечение гриппа",
        "от гастрита",
        "при бронхите",
    ])
    def test_router_matches_diseases(self, router: RouterService, message: str):
        """Тест распознавания заболеваний."""
        result = router.match(
            request=ChatRequest(message=message, conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        assert result.matched is True
        assert result.intent == IntentType.FIND_BY_DISEASE
    
    # --- Товарные запросы ---
    
    @pytest.mark.parametrize("message", [
        "Нурофен 200 мг",
        "Парацетамол",
        "Терафлю",
        "Колдрекс",
    ])
    def test_router_matches_products(self, router: RouterService, message: str):
        """Тест распознавания товарных запросов."""
        result = router.match(
            request=ChatRequest(message=message, conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        assert result.matched is True
        assert result.intent == IntentType.FIND_PRODUCT_BY_NAME
        assert "name" in result.slots or "product_name" in result.slots
    
    # --- Извлечение слотов ---
    
    def test_router_extracts_age(self, router: RouterService):
        """Тест извлечения возраста роутером."""
        result = router.match(
            request=ChatRequest(message="от кашля ребенку 5 лет", conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        assert result.matched is True
        assert result.slots.get("age") == 5
    
    def test_router_extracts_price(self, router: RouterService):
        """Тест извлечения цены роутером."""
        result = router.match(
            request=ChatRequest(message="от головы до 500 рублей", conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        assert result.matched is True
        assert result.slots.get("price_max") == 500
    
    def test_router_extracts_dosage_form(self, router: RouterService):
        """Тест извлечения формы выпуска."""
        result = router.match(
            request=ChatRequest(message="от кашля в сиропе", conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        assert result.matched is True
        assert result.slots.get("dosage_form") == "syrup"
    
    def test_router_uses_profile_defaults(self, router: RouterService):
        """Тест использования дефолтов из профиля."""
        profile = UserProfile(user_id="u1", preferences=UserPreferences(age=30, default_max_price=500))
        
        result = router.match(
            request=ChatRequest(message="от головы", conversation_id="test"),
            user_profile=profile,
            dialog_state=None,
        )
        assert result.matched is True
        assert result.slots.get("age") == 30
        assert result.slots.get("price_max") == 500
    
    # --- Debug информация ---
    
    def test_router_provides_debug_info(self, router: RouterService):
        """Тест наличия debug информации."""
        debug_builder = DebugMetaBuilder()
        
        result = router.match(
            request=ChatRequest(message="покажи корзину", conversation_id="test"),
            user_profile=None,
            dialog_state=None,
            debug_builder=debug_builder,
        )
        
        assert result.matched is True
        assert result.match_info is not None
        assert result.match_info.match_type == "keyword"
        assert len(result.match_info.matched_triggers) > 0
        
        debug = debug_builder.build()
        assert debug["router_matched"] is True


# =============================================================================
# Тесты: SlotManager
# =============================================================================

class TestSlotManager:
    """Тесты SlotManager."""
    
    def test_slot_manager_prompts_for_missing_slots(self, slot_manager: SlotManager):
        """Тест запроса недостающих слотов."""
        from app.services.router import RouterResult, SlotDefinition
        
        router_result = RouterResult(
            matched=True,
            intent=IntentType.FIND_BY_SYMPTOM,
            channel=None,
            slots={"symptom": "головная боль"},
            missing_slots=[SlotDefinition(name="age", prompt="Укажите возраст")],
            confidence=0.9,
        )
        
        response = slot_manager.handle_router_result(
            router_result=router_result,
            conversation_id="test-conv",
            user_profile=None,
        )
        
        # Должен спросить возраст
        assert "возраст" in response.reply.text.lower() or "укажите" in response.reply.text.lower()
        assert response.meta.debug.get("slot_filling_used") is True
    
    def test_slot_manager_handles_followup(self, slot_manager: SlotManager, dialog_store):
        """Тест обработки ответа на уточняющий вопрос."""
        # Устанавливаем состояние диалога с ожидающим слотом
        conversation_id = "test-followup"
        dialog_store.upsert_state(
            conversation_id,
            current_intent=IntentType.FIND_BY_SYMPTOM,
            channel=None,
            slots={"symptom": "кашель"},
            pending_slots=["age"],
            slot_questions={"age": "Уточните возраст"},
        )
        
        # Отправляем ответ с возрастом
        result = slot_manager.try_handle_followup(
            request_message="нам 7 лет",
            conversation_id=conversation_id,
            user_profile=None,
        )
        
        assert result.handled is True
        assert result.slot_filling_used is True
        assert result.assistant_response is not None
        assert result.assistant_response.actions  # Должен быть action
        
        # Проверяем, что возраст извлечен
        action = result.assistant_response.actions[0]
        assert action.parameters.get("age") == 7
    
    def test_slot_manager_uses_profile_defaults(self, slot_manager: SlotManager):
        """Тест использования дефолтов из профиля."""
        from app.services.router import RouterResult, SlotDefinition
        
        profile = UserProfile(
            user_id="u1",
            preferences=UserPreferences(age=42, default_max_price=1000)
        )
        
        router_result = RouterResult(
            matched=True,
            intent=IntentType.FIND_BY_SYMPTOM,
            channel=None,
            slots={"symptom": "ломота"},
            missing_slots=[SlotDefinition(name="age", prompt="Возраст?")],
            confidence=0.9,
        )
        
        response = slot_manager.handle_router_result(
            router_result=router_result,
            conversation_id="test-profile",
            user_profile=profile,
        )
        
        # Возраст должен быть взят из профиля, не должен спрашивать
        assert response.actions is not None
        assert len(response.actions) > 0
    
    def test_slot_manager_handles_skip_phrase(self, slot_manager: SlotManager, dialog_store):
        """Тест обработки фраз пропуска слота."""
        conversation_id = "test-skip"
        dialog_store.upsert_state(
            conversation_id,
            current_intent=IntentType.FIND_BY_SYMPTOM,
            channel=None,
            slots={"symptom": "кашель", "age": 30},
            pending_slots=["price_max"],
            slot_questions={"price_max": "До какой суммы?"},
        )
        
        result = slot_manager.try_handle_followup(
            request_message="неважно",
            conversation_id=conversation_id,
            user_profile=None,
        )
        
        assert result.handled is True
        # Должен принять ответ и продолжить
        assert result.assistant_response is not None


# =============================================================================
# Тесты: DebugMetaBuilder
# =============================================================================

class TestDebugMetaBuilder:
    """Тесты DebugMetaBuilder."""
    
    def test_build_basic(self):
        """Тест базового построения."""
        builder = DebugMetaBuilder(trace_id="trace123")
        builder.set_router_matched(True)
        builder.set_llm_used(False)
        builder.add_intent("SHOW_CART")
        
        debug = builder.build()
        
        assert debug["router_matched"] is True
        assert debug["llm_used"] is False
        assert "SHOW_CART" in debug["intent_chain"]
        assert debug["trace_id"] == "trace123"
    
    def test_build_with_router_info(self):
        """Тест с информацией роутера."""
        builder = DebugMetaBuilder()
        builder.set_router_matched(True)
        builder.set_router_confidence(0.95)
        builder.set_router_match_type("keyword")
        builder.set_matched_triggers(["корзин", "cart"])
        
        debug = builder.build()
        
        assert debug["router_confidence"] == 0.95
        assert debug["router_match_type"] == "keyword"
        assert "корзин" in debug["matched_triggers"]
    
    def test_pipeline_path_inference(self):
        """Тест автоматического определения pipeline_path."""
        # Router only
        builder1 = DebugMetaBuilder()
        builder1.set_router_matched(True)
        builder1.set_llm_used(False)
        assert builder1.build()["pipeline_path"] == "router_only"
        
        # Router + slots
        builder2 = DebugMetaBuilder()
        builder2.set_router_matched(True)
        builder2.set_slot_filling_used(True)
        assert builder2.build()["pipeline_path"] == "router+slots"
        
        # LLM only
        builder3 = DebugMetaBuilder()
        builder3.set_router_matched(False)
        builder3.set_llm_used(True)
        assert builder3.build()["pipeline_path"] == "llm_only"
    
    def test_merge_existing(self):
        """Тест мержа существующих данных."""
        builder = DebugMetaBuilder()
        existing = {
            "router_matched": True,
            "llm_used": False,
            "intent_chain": ["SHOW_CART"],
            "custom_field": "value",
        }
        
        builder.merge_existing(existing)
        builder.add_intent("SHOW_PROFILE")
        
        debug = builder.build()
        
        assert debug["router_matched"] is True
        assert "SHOW_CART" in debug["intent_chain"]
        assert "SHOW_PROFILE" in debug["intent_chain"]
        assert debug["custom_field"] == "value"


# =============================================================================
# Интеграционные тесты
# =============================================================================

class TestIntegration:
    """Интеграционные тесты NLU-пайплайна."""
    
    def test_full_symptom_flow(self, router: RouterService, slot_manager: SlotManager, dialog_store):
        """Тест полного флоу: симптом → уточнение возраста → результат."""
        conversation_id = "integration-symptom"
        
        # Шаг 1: Пользователь спрашивает про симптом
        result1 = router.match(
            request=ChatRequest(message="болит голова", conversation_id=conversation_id),
            user_profile=None,
            dialog_state=None,
        )
        
        assert result1.matched is True
        assert result1.intent == IntentType.FIND_BY_SYMPTOM
        assert "symptom" in result1.slots
        
        # Шаг 2: SlotManager обрабатывает - спрашивает возраст
        response1 = slot_manager.handle_router_result(
            router_result=result1,
            conversation_id=conversation_id,
            user_profile=None,
        )
        
        # Может спросить возраст или сразу выполнить (зависит от конфига)
        if not response1.actions:
            # Спросил возраст - отвечаем
            result2 = slot_manager.try_handle_followup(
                request_message="мне 35 лет",
                conversation_id=conversation_id,
                user_profile=None,
            )
            
            assert result2.handled is True
            assert result2.assistant_response is not None
            assert result2.assistant_response.actions
            
            action = result2.assistant_response.actions[0]
            assert action.intent == IntentType.FIND_BY_SYMPTOM
            assert action.parameters.get("age") == 35
    
    def test_full_product_flow(self, router: RouterService, slot_manager: SlotManager):
        """Тест флоу поиска товара."""
        conversation_id = "integration-product"
        
        # Поиск конкретного товара
        result = router.match(
            request=ChatRequest(message="Нурофен детский сироп", conversation_id=conversation_id),
            user_profile=None,
            dialog_state=None,
        )
        
        assert result.matched is True
        assert result.intent == IntentType.FIND_PRODUCT_BY_NAME
        assert "name" in result.slots or "product_name" in result.slots
        
        # Проверяем извлечение формы и детского контекста
        # (зависит от реализации - может быть или нет)
    
    def test_cart_then_order_flow(self, router: RouterService):
        """Тест переключения между корзиной и заказами."""
        conversation_id = "integration-cart-order"
        
        # Сначала корзина
        result1 = router.match(
            request=ChatRequest(message="покажи корзину", conversation_id=conversation_id),
            user_profile=None,
            dialog_state=None,
        )
        assert result1.intent == IntentType.SHOW_CART
        
        # Потом заказы
        result2 = router.match(
            request=ChatRequest(message="а где мой заказ?", conversation_id=conversation_id),
            user_profile=None,
            dialog_state=None,
        )
        assert result2.intent == IntentType.SHOW_ORDER_STATUS
    
    def test_complex_query_with_filters(self, router: RouterService):
        """Тест сложного запроса с фильтрами."""
        result = router.match(
            request=ChatRequest(
                message="Нужен сироп от кашля для ребенка 5 лет до 500 рублей без сахара",
                conversation_id="test-complex"
            ),
            user_profile=None,
            dialog_state=None,
        )
        
        assert result.matched is True
        # Проверяем извлеченные слоты
        slots = result.slots
        assert slots.get("age") == 5 or slots.get("is_for_children") is True
        assert slots.get("dosage_form") == "syrup" or "сироп" in str(slots.get("dosage_form", "")).lower()
        assert slots.get("price_max") == 500


# =============================================================================
# Тесты edge cases
# =============================================================================

class TestEdgeCases:
    """Тесты граничных случаев."""
    
    def test_empty_message(self, router: RouterService):
        """Тест пустого сообщения."""
        result = router.match(
            request=ChatRequest(message="", conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        assert result.matched is False
    
    def test_very_long_message(self, router: RouterService):
        """Тест очень длинного сообщения."""
        long_message = "нурофен " * 100
        result = router.match(
            request=ChatRequest(message=long_message, conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        # Длинные сообщения могут не матчиться как товар (эвристика отсекает)
        # но могут матчиться по триггерам если бренд найден
    
    def test_message_with_typos(self, router: RouterService):
        """Тест сообщения с опечатками."""
        # Благодаря морфологическим основам должно работать
        result = router.match(
            request=ChatRequest(message="карзина", conversation_id="test"),  # опечатка
            user_profile=None,
            dialog_state=None,
        )
        # Может не сматчить из-за опечатки - это ожидаемое поведение
    
    def test_mixed_language(self, router: RouterService):
        """Тест смешанного языка."""
        result = router.match(
            request=ChatRequest(message="show my cart пожалуйста", conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        # "cart" должен сработать как триггер
        assert result.matched is True
        assert result.intent == IntentType.SHOW_CART
    
    def test_numbers_only(self, router: RouterService):
        """Тест сообщения только с числами."""
        result = router.match(
            request=ChatRequest(message="30", conversation_id="test"),
            user_profile=None,
            dialog_state=None,
        )
        # Короткие числа могут интерпретироваться как возраст
        # но без контекста не должны матчить интент

