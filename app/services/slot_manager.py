"""
SlotManager - Управление слотами и диалоговое уточнение.

Ключевые функции:
1. Определение недостающих слотов для интента
2. Формирование уточняющих вопросов
3. Извлечение слотов из ответов пользователя (без LLM)
4. Сохранение состояния диалога
5. Применение дефолтов из профиля

Поддерживаемые слоты:
- symptom (симптом)
- disease (заболевание)
- age (возраст)
- is_for_children (детский препарат)
- is_pregnant (беременность)
- price_min, price_max (ценовой диапазон)
- price_filter_disabled (цена неважна)
- dosage_form (форма выпуска)
- is_otc (без рецепта)
- delivery_type (способ доставки)
- pharmacy_id (конкретная аптека)
- product_id (товар)
- order_id (заказ)
- metro (станция метро)
- promo_code (промокод)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from langsmith import traceable
import yaml

from ..intents import ActionChannel, ActionType, IntentType
from ..models.assistant import AssistantAction, AssistantMeta, AssistantResponse, Reply
from .dialog_state_store import DialogStateStore, get_dialog_state_store
from .debug_meta import DebugMetaBuilder
from .response_helpers import slot_quick_replies
from .router import CONFIG_PATH, RouterResult
from .slot_extraction import (
    extract_age,
    extract_age_group,
    age_to_age_group,
    AgeGroup,
    PregnancyStatus,
    extract_price_max,
    extract_price_min,
    extract_dosage_form,
    extract_dosage_forms,
    extract_symptom,
    extract_symptoms,
    extract_disease,
    extract_symptom_duration,
    extract_symptom_severity,
    extract_is_for_children,
    extract_is_for_teenager,
    extract_is_for_elderly,
    extract_special_filters,
    extract_pregnancy_context,
    extract_pregnancy_status,
    extract_chronic_conditions,
    extract_allergies,
    extract_otc_preference,
    check_price_indifference,
    extract_all_entities,
    normalize_text,
)
from .user_profile_store import UserProfile, UserProfileStore, get_user_profile_store

logger = logging.getLogger(__name__)


# =============================================================================
# Конфигурация слотов
# =============================================================================

# Приоритет слотов для уточнения (меньше = спрашивать раньше)
SLOT_PRIORITY: Dict[str, int] = {
    # Основная информация о запросе
    "symptom": 1,
    "disease": 1,
    "product_id": 1,
    "product_name": 1,
    "order_id": 2,
    # Критичные для безопасности
    "age_group": 3,  # Возрастная группа - критичный слот для подбора лекарств
    "age": 4,
    "pregnancy_status": 5,  # Критично для женщин детородного возраста
    # Дополнительная медицинская информация
    "chronic_conditions": 6,
    "has_chronic_conditions": 6,
    "has_allergies": 7,
    "allergies": 7,
    "current_medications": 8,
    # Характеристики симптомов
    "symptom_severity": 9,
    "symptom_duration": 9,
    # Предпочтения
    "dosage_form": 10,
    "price_max": 11,
    # Логистика
    "pharmacy_id": 12,
    "metro": 12,
    "promo_code": 13,
    "delivery_type": 14,
}

# Слоты, которые можно вывести из контекста
CONTEXTUAL_SLOTS: Set[str] = {
    "age",
    "price_max",
    "dosage_form",
    "is_for_children",
    "sugar_free",
    "lactose_free",
}

# Интенты, для которых возраст/возрастная группа КРИТИЧНЫ (обязательный слот)
AGE_CRITICAL_INTENTS: Set[IntentType] = {
    IntentType.FIND_BY_SYMPTOM,
    IntentType.SYMPTOM_TO_PRODUCT,
    IntentType.FIND_BY_DISEASE,
    IntentType.DISEASE_TO_PRODUCT,
}

# Интенты, для которых возраст опционален
AGE_OPTIONAL_INTENTS: Set[IntentType] = {
    IntentType.FIND_BY_CATEGORY,
    IntentType.FIND_PRODUCT_BY_NAME,
    IntentType.FIND_POPULAR,
    IntentType.FIND_PROMO,
}

# Интенты, требующие возрастную группу (age_group)
AGE_GROUP_REQUIRED_INTENTS: Set[IntentType] = {
    IntentType.FIND_BY_SYMPTOM,
    IntentType.SYMPTOM_TO_PRODUCT,
    IntentType.FIND_BY_DISEASE,
    IntentType.DISEASE_TO_PRODUCT,
}

# Интенты, для которых важен статус беременности (для женщин детородного возраста)
PREGNANCY_RELEVANT_INTENTS: Set[IntentType] = {
    IntentType.FIND_BY_SYMPTOM,
    IntentType.SYMPTOM_TO_PRODUCT,
    IntentType.FIND_BY_DISEASE,
    IntentType.DISEASE_TO_PRODUCT,
}

# Интенты, для которых важны хронические заболевания
CHRONIC_CONDITIONS_RELEVANT_INTENTS: Set[IntentType] = {
    IntentType.FIND_BY_SYMPTOM,
    IntentType.SYMPTOM_TO_PRODUCT,
    IntentType.FIND_BY_DISEASE,
    IntentType.DISEASE_TO_PRODUCT,
}

# Фразы для распознавания отказа от уточнения
SKIP_PHRASES: Set[str] = {
    "неважно", "не важно", "любой", "любая", "любое",
    "все равно", "всё равно", "без разницы", "пропустить",
    "пропущу", "не знаю", "незнаю", "сколько угодно",
    "на ваш выбор", "на ваше усмотрение", "какой есть",
}

# Вопросы для слотов по умолчанию (будут переопределены из конфига)
DEFAULT_SLOT_QUESTIONS: Dict[str, str] = {
    # Основные медицинские
    "symptom": "Какой симптом беспокоит? (головная боль, кашель, насморк, температура...)",
    "disease": "Какое заболевание или диагноз?",
    "symptom_duration": "Как давно беспокоит этот симптом? (сегодня, несколько дней, неделю...)",
    "symptom_severity": "Насколько сильно выражен симптом: слабо, умеренно или сильно?",
    # Возраст и особые категории
    "age_group": "Для кого подбираем препарат: взрослый, ребёнок, подросток или пожилой человек?",
    "age": "Сколько лет? Это важно для подбора безопасной дозировки.",
    "pregnancy_status": "Уточните, пожалуйста: беременность, кормление грудью или ни то, ни другое?",
    # Противопоказания и безопасность
    "has_chronic_conditions": "Есть ли хронические заболевания? (диабет, гипертония, астма, проблемы с ЖКТ...)",
    "chronic_conditions": "Какие именно хронические заболевания?",
    "has_allergies": "Есть ли аллергия на какие-либо лекарства или компоненты?",
    "allergies": "На какие именно препараты или вещества аллергия?",
    "current_medications": "Принимаете ли сейчас какие-то лекарства на постоянной основе?",
    # Предпочтения по препарату
    "price_max": "До какой суммы смотреть варианты?",
    "dosage_form": "В какой форме предпочитаете? (таблетки, сироп, спрей, капли...)",
    "brand_preference": "Предпочитаете оригинальные препараты или подойдут аналоги?",
    # Товары и заказы
    "product_id": "Какой товар имеете в виду? Укажите название.",
    "product_name": "Как называется препарат?",
    "order_id": "Укажите номер заказа.",
    # Логистика
    "pharmacy_id": "Какую аптеку выбираете?",
    "metro": "Возле какой станции метро ищем?",
    "promo_code": "Введите промокод.",
    "delivery_type": "Как хотите получить заказ? (самовывоз или доставка)",
}


@dataclass
class SlotHandlingResult:
    """Результат обработки слотов."""
    handled: bool
    assistant_response: Optional[AssistantResponse] = None
    slot_filling_used: bool = False
    filled_slots: Dict[str, Any] = None
    pending_slots: List[str] = None
    
    def __post_init__(self):
        if self.filled_slots is None:
            self.filled_slots = {}
        if self.pending_slots is None:
            self.pending_slots = []


class SlotManager:
    """
    Менеджер слотов для диалогового уточнения.
    
    Логика:
    1. Если есть незакрытые слоты в состоянии диалога → пытаемся закрыть
    2. Если Router вернул интент с недостающими слотами → спрашиваем
    3. Если все слоты заполнены → возвращаем action для платформы
    """

    def __init__(
        self,
        dialog_state_store: DialogStateStore | None = None,
        user_profile_store: UserProfileStore | None = None,
        config_path: Path | None = None,
    ) -> None:
        self._dialog_state_store = dialog_state_store or get_dialog_state_store()
        self._user_profile_store = user_profile_store or get_user_profile_store()
        self._config_path = config_path or CONFIG_PATH
        
        config = self._load_config(self._config_path)
        self._intent_slot_questions = self._build_slot_questions(config)
        self._dosage_form_keywords = self._load_dosage_forms(config)
        self._default_slot_questions = {
            **DEFAULT_SLOT_QUESTIONS,
            **((config.get("defaults") or {}).get("slot_questions") or {}),
        }

    # =========================================================================
    # Публичные методы
    # =========================================================================

    @traceable(run_type="chain", name="slot_manager_followup")
    def try_handle_followup(
        self,
        *,
        request_message: str,
        conversation_id: str,
        user_profile: UserProfile | None,
        debug_builder: DebugMetaBuilder | None = None,
        trace_id: str | None = None,
    ) -> SlotHandlingResult:
        """
        Пытается обработать сообщение как ответ на уточняющий вопрос.
        
        Вызывается ПЕРВЫМ в пайплайне, до Router'а.
        
        Returns:
            SlotHandlingResult с результатом обработки
        """
        state = self._dialog_state_store.get_state(conversation_id)
        if not state or not state.pending_slots:
            return SlotHandlingResult(handled=False)
        
        logger.debug(
            "trace_id=%s Handling followup for intent=%s pending_slots=%s",
            trace_id or "-",
            state.current_intent,
            state.pending_slots,
        )
        
        # Извлекаем слоты из ответа
        extracted = self._extract_slots_from_response(request_message, state.pending_slots, user_profile)
        
        # Проверяем, не хочет ли пользователь пропустить
        normalized = normalize_text(request_message)
        wants_skip = any(phrase in normalized for phrase in SKIP_PHRASES)
        
        # Мержим слоты
        merged_slots = {**state.slots}
        for key, value in extracted.items():
            if value is not None:
                merged_slots[key] = value
        
        # Применяем дефолты из профиля
        merged_slots = self._apply_profile_defaults(merged_slots, user_profile)
        
        # Определяем оставшиеся незаполненные слоты
        remaining = []
        for slot in state.pending_slots:
            slot_value = merged_slots.get(slot)
            
            # Слот заполнен
            if slot_value is not None and slot_value != "" and slot_value != []:
                continue
            
            # Пользователь хочет пропустить этот слот
            if wants_skip:
                # Помечаем как "неважно" для ценовых слотов
                if slot in ("price_max", "price_min"):
                    merged_slots["price_filter_disabled"] = True
                continue
            
            remaining.append(slot)
        
        # Если остались незаполненные слоты - спрашиваем следующий
        if remaining:
            next_slot = self._get_next_slot_to_ask(remaining)
            prompt = state.slot_questions.get(next_slot) or self._question_for_slot(state.current_intent, next_slot)
            
            self._dialog_state_store.upsert_state(
                conversation_id,
                slots=merged_slots,
                pending_slots=remaining,
                slot_questions=state.slot_questions,
                last_prompt=prompt,
            )
            
            response = self._build_prompt_response(
                intent=state.current_intent,
                prompt=prompt,
                filled_slots=merged_slots,
                pending_slots=remaining,
            )
            
            if debug_builder:
                debug_builder.set_slot_filling_used(True)
                debug_builder.set_pending_slots(True)
                debug_builder.add_intent(getattr(state.current_intent, "value", None))
                debug_builder.add_extra("filled_slots", list(merged_slots.keys()))
                debug_builder.add_extra("remaining_slots", remaining)
            
            logger.info(
                "trace_id=%s intent=%s slot_followup: filled=%s remaining=%s",
                trace_id or "-",
                state.current_intent,
                list(extracted.keys()),
                remaining,
            )
            
            return SlotHandlingResult(
                handled=True,
                assistant_response=response,
                slot_filling_used=True,
                filled_slots=merged_slots,
                pending_slots=remaining,
            )
        
        # Все слоты заполнены - формируем action
        self._dialog_state_store.clear_state(conversation_id)
        
        action_response = self._build_action_response(
            intent=state.current_intent,
            channel=state.channel,
            parameters=merged_slots,
            slot_filling_used=True,
        )
        
        if debug_builder:
            debug_builder.set_slot_filling_used(True)
            debug_builder.set_pending_slots(False)
            debug_builder.add_intent(getattr(state.current_intent, "value", None))
            debug_builder.add_extra("final_slots", merged_slots)
        
        logger.info(
            "trace_id=%s intent=%s slot_followup completed: final_slots=%s",
            trace_id or "-",
            state.current_intent,
            list(merged_slots.keys()),
        )
        
        return SlotHandlingResult(
            handled=True,
            assistant_response=action_response,
            slot_filling_used=True,
            filled_slots=merged_slots,
            pending_slots=[],
        )

    @traceable(run_type="chain", name="slot_manager_handle_router_result")
    def handle_router_result(
        self,
        *,
        router_result: RouterResult,
        conversation_id: str,
        user_profile: UserProfile | None = None,
        debug_builder: DebugMetaBuilder | None = None,
        trace_id: str | None = None,
    ) -> AssistantResponse:
        """
        Обрабатывает результат Router'а.
        
        Если есть недостающие слоты - спрашивает.
        Если все слоты есть - возвращает action.
        
        Returns:
            AssistantResponse с промптом или action
        """
        # Применяем дефолты из профиля
        slots = self._apply_profile_defaults(dict(router_result.slots), user_profile)
        
        # Получаем вопросы для слотов
        slot_questions = router_result.slot_questions or self._intent_slot_questions.get(router_result.intent, {})
        
        # Определяем реально недостающие слоты (с учетом профиля)
        pending_slots = []
        for slot_def in router_result.missing_slots:
            slot_name = slot_def.name
            slot_value = slots.get(slot_name)
            
            if slot_value is None or slot_value == "" or slot_value == []:
                # Проверяем, критичен ли этот слот для интента
                if self._is_slot_required(router_result.intent, slot_name, slots):
                    pending_slots.append(slot_name)
        
        # Если есть недостающие слоты - сохраняем состояние и спрашиваем
        if pending_slots:
            next_slot = self._get_next_slot_to_ask(pending_slots)
            prompt = self._get_slot_prompt(router_result, next_slot, slot_questions)
            
            self._dialog_state_store.upsert_state(
                conversation_id,
                current_intent=router_result.intent,
                channel=router_result.channel,
                slots=slots,
                pending_slots=pending_slots,
                slot_questions=slot_questions,
                last_prompt=prompt,
            )
            
            if debug_builder:
                debug_builder.set_slot_filling_used(True)
                debug_builder.set_pending_slots(True)
                debug_builder.add_intent(getattr(router_result.intent, "value", None))
                debug_builder.add_extra("pending_slots", pending_slots)
            
            logger.info(
                "trace_id=%s intent=%s slots_required pending=%s",
                trace_id or "-",
                router_result.intent,
                pending_slots,
            )
            
            return self._build_prompt_response(
                intent=router_result.intent,
                prompt=prompt,
                filled_slots=slots,
                pending_slots=pending_slots,
                confidence=router_result.confidence,
            )
        
        # Все слоты заполнены - очищаем состояние и возвращаем action
        self._dialog_state_store.clear_state(conversation_id)
        
        if debug_builder:
            debug_builder.set_slot_filling_used(False)
            debug_builder.set_pending_slots(False)
            debug_builder.add_intent(getattr(router_result.intent, "value", None))
        
        logger.info(
            "trace_id=%s intent=%s slots_complete slots=%s",
            trace_id or "-",
            router_result.intent,
            list(slots.keys()),
        )
        
        return self._build_action_response(
            intent=router_result.intent,
            channel=router_result.channel,
            parameters=slots,
            slot_filling_used=False,
        )

    # =========================================================================
    # Приватные методы - извлечение слотов
    # =========================================================================

    def _extract_slots_from_response(
        self,
        message: str,
        pending_slots: List[str],
        user_profile: UserProfile | None,
    ) -> Dict[str, Any]:
        """
        Извлекает слоты из ответа пользователя.
        
        Фокусируется на pending_slots, но извлекает все найденные сущности.
        """
        # Используем комплексное извлечение
        extraction = extract_all_entities(message, self._dosage_form_keywords)
        slots = extraction.to_slots_dict()
        
        # Дополнительно проверяем специфичные слоты
        normalized = normalize_text(message)
        
        # Возрастная группа - критичный слот
        if "age_group" in pending_slots and not slots.get("age_group"):
            age_group = extract_age_group(message)
            if age_group:
                slots["age_group"] = age_group
            # Если извлекли возраст, но не группу - преобразуем
            elif slots.get("age"):
                slots["age_group"] = age_to_age_group(slots["age"])
            # Проверяем контекстные маркеры
            elif extract_is_for_children(message):
                slots["age_group"] = AgeGroup.CHILD
                slots["is_for_children"] = True
            elif extract_is_for_teenager(message):
                slots["age_group"] = AgeGroup.TEENAGER
                slots["is_for_teenager"] = True
            elif extract_is_for_elderly(message):
                slots["age_group"] = AgeGroup.ELDERLY
                slots["is_for_elderly"] = True
            # "взрослый" или просто короткий ответ без детских маркеров
            elif "взросл" in normalized or "мне" in normalized or "себе" in normalized:
                slots["age_group"] = AgeGroup.ADULT
        
        # Симптомы
        if "symptom" in pending_slots and not slots.get("symptom"):
            symptoms = extract_symptoms(message)
            if symptoms:
                slots["symptom"] = symptoms[0]
                if len(symptoms) > 1:
                    slots["symptoms"] = symptoms
        
        # Заболевания
        if "disease" in pending_slots and not slots.get("disease"):
            disease = extract_disease(message)
            if disease:
                slots["disease"] = disease
        
        # Статус беременности
        if "pregnancy_status" in pending_slots and not slots.get("pregnancy_status"):
            pregnancy_status = extract_pregnancy_status(message)
            if pregnancy_status:
                slots["pregnancy_status"] = pregnancy_status
        
        # Хронические заболевания
        if "chronic_conditions" in pending_slots or "has_chronic_conditions" in pending_slots:
            has_chronic, conditions = extract_chronic_conditions(message)
            if has_chronic is not None:
                slots["has_chronic_conditions"] = has_chronic
                if conditions:
                    slots["chronic_conditions"] = conditions
        
        # Аллергии
        if "allergies" in pending_slots or "has_allergies" in pending_slots:
            has_allergies, allergens = extract_allergies(message)
            if has_allergies is not None:
                slots["has_allergies"] = has_allergies
                if allergens:
                    slots["allergies"] = allergens
        
        # Длительность симптомов
        if "symptom_duration" in pending_slots and not slots.get("symptom_duration"):
            duration = extract_symptom_duration(message)
            if duration:
                slots["symptom_duration"] = duration
        
        # Выраженность симптомов
        if "symptom_severity" in pending_slots and not slots.get("symptom_severity"):
            severity = extract_symptom_severity(message)
            if severity:
                slots["symptom_severity"] = severity
        
        # Метро (простое извлечение)
        if "metro" in pending_slots:
            metro = self._extract_metro(message)
            if metro:
                slots["metro"] = metro
        
        # Промокод (обычно короткий код)
        if "promo_code" in pending_slots:
            promo = self._extract_promo_code(message)
            if promo:
                slots["promo_code"] = promo
        
        # Номер заказа
        if "order_id" in pending_slots:
            order_id = self._extract_order_id(message)
            if order_id:
                slots["order_id"] = order_id
        
        # Product name/id - берем весь текст если он похож на название
        if "product_id" in pending_slots or "product_name" in pending_slots:
            if len(message.split()) <= 5 and not self._is_generic_response(message):
                slots["product_name"] = message.strip()
                slots["name"] = message.strip()
        
        # Способ доставки
        if "delivery_type" in pending_slots:
            delivery = self._extract_delivery_type(normalized)
            if delivery:
                slots["delivery_type"] = delivery
        
        # Обновляем профиль если извлекли важные данные
        if user_profile:
            self._update_profile_from_slots(user_profile.user_id, slots)
        
        return slots

    def _extract_metro(self, message: str) -> Optional[str]:
        """Извлекает название станции метро."""
        import re
        
        patterns = [
            r"(?:у\s+)?метро\s+([а-яё\s]+)",
            r"(?:станци[яи]|ст\.?)\s+([а-яё\s]+)",
            r"возле\s+([а-яё\s]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                metro = match.group(1).strip()
                # Убираем стоп-слова
                for stop in ["метро", "станция", "станции"]:
                    metro = metro.replace(stop, "").strip()
                if metro:
                    return metro.title()
        
        # Если короткий ответ - возможно это просто название станции
        if len(message.split()) <= 3:
            return message.strip().title()
        
        return None

    def _extract_promo_code(self, message: str) -> Optional[str]:
        """Извлекает промокод из сообщения."""
        import re
        
        # Промокоды обычно в верхнем регистре и содержат буквы/цифры
        match = re.search(r"([A-ZА-ЯЁ0-9]{3,15})", message.upper())
        if match:
            return match.group(1)
        
        # Или просто берем короткое сообщение
        if len(message.split()) == 1:
            return message.strip().upper()
        
        return None

    def _extract_order_id(self, message: str) -> Optional[str]:
        """Извлекает номер заказа."""
        import re
        
        # Номер заказа обычно содержит цифры
        match = re.search(r"№?\s*(\d{4,})", message)
        if match:
            return match.group(1)
        
        # Или формат типа ORD-12345
        match = re.search(r"([A-ZА-ЯЁ]{2,5}[-\s]?\d{3,})", message.upper())
        if match:
            return match.group(1)
        
        return None

    def _extract_delivery_type(self, normalized: str) -> Optional[str]:
        """Извлекает способ доставки."""
        pickup_markers = ["самовывоз", "заберу сам", "сам заберу", "в аптеке", "в аптеку"]
        delivery_markers = ["курьер", "доставка", "на дом", "домой", "привезите", "привезти"]
        
        if any(marker in normalized for marker in pickup_markers):
            return "pickup"
        if any(marker in normalized for marker in delivery_markers):
            return "delivery"
        
        return None

    def _is_generic_response(self, message: str) -> bool:
        """Проверяет, является ли ответ общим (не содержит конкретной информации)."""
        normalized = normalize_text(message)
        generic = SKIP_PHRASES | {"да", "нет", "ок", "хорошо", "понятно", "ладно"}
        return normalized in generic

    # =========================================================================
    # Приватные методы - логика слотов
    # =========================================================================

    def _is_slot_required(
        self,
        intent: IntentType | None,
        slot_name: str,
        current_slots: Dict[str, Any],
    ) -> bool:
        """
        Определяет, действительно ли нужен слот для данного интента.
        
        Некоторые слоты могут быть опциональными в зависимости от контекста.
        """
        if not intent:
            return True
        
        # age_group ОБЯЗАТЕЛЕН для критичных интентов (подбор лекарств)
        if slot_name == "age_group":
            # Для интентов подбора по симптомам/болезням - ОБЯЗАТЕЛЬНО
            if intent in AGE_GROUP_REQUIRED_INTENTS:
                # Если уже есть age_group - не нужен
                if current_slots.get("age_group"):
                    return False
                # Если есть числовой возраст - можно вычислить age_group
                if current_slots.get("age"):
                    return False
                # Если есть детский/подростковый/пожилой контекст
                if current_slots.get("is_for_children") or current_slots.get("is_for_teenager") or current_slots.get("is_for_elderly") or current_slots.get("is_for_adults"):
                    return False
                return True
            return False  # Для остальных интентов - опционален
        
        # Возраст (точный) опционален - достаточно age_group
        if slot_name == "age":
            # Если есть age_group - точный возраст не обязателен
            if current_slots.get("age_group"):
                return False
            if intent in AGE_OPTIONAL_INTENTS:
                return False
            # Если указан детский контекст, можем не спрашивать точный возраст
            if current_slots.get("is_for_children") or current_slots.get("is_for_teenager") or current_slots.get("is_for_elderly"):
                return False
            return False  # age не обязателен, если есть age_group
        
        # Цена всегда опциональна
        if slot_name in ("price_max", "price_min"):
            return False
        
        # dosage_form опционален
        if slot_name == "dosage_form":
            return False
        
        return True

    def _get_next_slot_to_ask(self, pending_slots: List[str]) -> str:
        """Выбирает следующий слот для уточнения по приоритету."""
        if not pending_slots:
            return pending_slots[0] if pending_slots else ""
        
        # Сортируем по приоритету
        sorted_slots = sorted(
            pending_slots,
            key=lambda s: SLOT_PRIORITY.get(s, 10),
        )
        
        return sorted_slots[0]

    def _get_slot_prompt(
        self,
        router_result: RouterResult,
        slot_name: str,
        slot_questions: Dict[str, str],
    ) -> str:
        """Получает промпт для конкретного слота."""
        # Сначала из slot_questions роутера
        if slot_name in slot_questions:
            return slot_questions[slot_name]
        
        # Затем из missing_slots роутера
        for slot_def in router_result.missing_slots:
            if slot_def.name == slot_name:
                return slot_def.prompt
        
        # Fallback на вопрос по интенту
        return self._question_for_slot(router_result.intent, slot_name)

    def _question_for_slot(self, intent: IntentType | None, slot: str) -> str:
        """Возвращает вопрос для слота с учетом интента."""
        # Сначала проверяем вопросы для конкретного интента
        if intent and intent in self._intent_slot_questions:
            question = self._intent_slot_questions[intent].get(slot)
            if question:
                return question
        
        # Fallback на дефолтный вопрос
        return self._default_slot_questions.get(slot, "Уточните, пожалуйста.")

    def _apply_profile_defaults(
        self,
        slots: Dict[str, Any],
        user_profile: UserProfile | None,
    ) -> Dict[str, Any]:
        """Применяет дефолты из профиля пользователя."""
        if not user_profile:
            return slots
        
        prefs = user_profile.preferences
        
        # Возраст и возрастная группа
        profile_age = getattr(prefs, "age", None)
        if profile_age:
            if not slots.get("age"):
                slots["age"] = profile_age
            # Если нет age_group, но есть возраст - вычисляем группу
            if not slots.get("age_group"):
                slots["age_group"] = age_to_age_group(profile_age)
        
        # Возрастная группа из профиля (если явно указана)
        profile_age_group = getattr(prefs, "age_group", None)
        if profile_age_group and not slots.get("age_group"):
            slots["age_group"] = profile_age_group
        
        # Цена
        default_price = getattr(prefs, "default_max_price", None)
        if not slots.get("price_max") and default_price is not None:
            slots["price_max"] = default_price
        
        # Формы выпуска
        preferred_forms = getattr(prefs, "preferred_forms", None) or []
        if not slots.get("dosage_form") and preferred_forms:
            slots["dosage_form"] = preferred_forms[0]
        
        # Булевые предпочтения
        if getattr(prefs, "sugar_free", False):
            slots.setdefault("sugar_free", True)
        if getattr(prefs, "lactose_free", False):
            slots.setdefault("lactose_free", True)
        if getattr(prefs, "for_children", False):
            slots.setdefault("is_for_children", True)
            if not slots.get("age_group"):
                slots["age_group"] = AgeGroup.CHILD
        
        # Хронические заболевания из профиля
        profile_chronic = getattr(prefs, "chronic_conditions", None)
        if profile_chronic and not slots.get("chronic_conditions"):
            slots["chronic_conditions"] = profile_chronic
            slots["has_chronic_conditions"] = True
        
        # Аллергии из профиля
        profile_allergies = getattr(prefs, "allergies", None)
        if profile_allergies and not slots.get("allergies"):
            slots["allergies"] = profile_allergies
            slots["has_allergies"] = True
        
        return slots

    def _update_profile_from_slots(self, user_id: str, slots: Dict[str, Any]) -> None:
        """Обновляет профиль пользователя из извлеченных слотов."""
        updates: Dict[str, Any] = {}
        
        if slots.get("age"):
            updates["age"] = slots["age"]
        
        if slots.get("price_max"):
            updates["default_max_price"] = slots["price_max"]
        
        if slots.get("dosage_form"):
            existing_forms = []
            profile = self._user_profile_store.get_or_create(user_id)
            if profile and profile.preferences:
                existing_forms = profile.preferences.preferred_forms or []
            
            form = slots["dosage_form"]
            if form not in existing_forms:
                existing_forms = [form] + list(existing_forms)
                updates["preferred_forms"] = existing_forms[:5]  # Ограничиваем
        
        if slots.get("sugar_free"):
            updates["sugar_free"] = True
        if slots.get("is_for_children"):
            updates["for_children"] = True
        
        if updates:
            self._user_profile_store.update_preferences(user_id, **updates)

    # =========================================================================
    # Приватные методы - построение ответов
    # =========================================================================

    def _build_prompt_response(
        self,
        intent: IntentType | None,
        prompt: str,
        filled_slots: Dict[str, Any],
        pending_slots: List[str],
        confidence: float = 0.9,
    ) -> AssistantResponse:
        """Строит ответ с уточняющим вопросом и кнопками быстрого ответа."""
        next_slot = pending_slots[0] if pending_slots else None
        
        # Получаем варианты быстрого ответа для текущего слота
        quick_replies = slot_quick_replies(next_slot, intent) if next_slot else []
        
        meta = AssistantMeta(
            top_intent=getattr(intent, "value", None),
            confidence=confidence,
            quick_replies=quick_replies if quick_replies else None,
            debug={
                "slot_filling_used": True,
                "slot_prompt_pending": True,
                "filled_slots": list(filled_slots.keys()),
                "pending_slots": pending_slots,
                "final_slots": filled_slots,  # Текущее состояние слотов
                "next_slot": next_slot,
            },
        )
        
        return AssistantResponse(
            reply=Reply(text=prompt),
            actions=[],
            meta=meta,
        )

    def _build_action_response(
        self,
        intent: IntentType | None,
        channel: ActionChannel | None,
        parameters: Dict[str, Any],
        slot_filling_used: bool,
    ) -> AssistantResponse:
        """Строит ответ с action для платформы."""
        action = AssistantAction(
            type=ActionType.CALL_PLATFORM_API,
            intent=intent,
            channel=channel,
            parameters=parameters,
        )
        
        meta = AssistantMeta(
            top_intent=getattr(intent, "value", None),
            confidence=0.95,
            debug={
                "slot_filling_used": slot_filling_used,
                "slot_prompt_pending": False,
                "pending_slots": [],  # Все слоты заполнены
                "filled_slots": list(parameters.keys()),
                "final_slots": parameters,  # Полный набор слотов
            },
        )
        
        # Формируем текст ответа в зависимости от интента
        reply_text = self._get_action_reply_text(intent, parameters)
        
        return AssistantResponse(
            reply=Reply(text=reply_text),
            actions=[action],
            meta=meta,
        )

    def _get_action_reply_text(self, intent: IntentType | None, parameters: Dict[str, Any]) -> str:
        """Генерирует текст ответа для action."""
        if not intent:
            return "Понял, выполняю запрос."
        
        intent_replies = {
            IntentType.FIND_BY_SYMPTOM: "Подбираю препараты по вашему запросу.",
            IntentType.FIND_BY_DISEASE: "Ищу подходящие препараты.",
            IntentType.FIND_PRODUCT_BY_NAME: f"Ищу «{parameters.get('name', parameters.get('product_name', 'товар'))}».",
            IntentType.SHOW_CART: "Показываю вашу корзину.",
            IntentType.SHOW_NEARBY_PHARMACIES: "Ищу ближайшие аптеки.",
            IntentType.SHOW_ACTIVE_ORDERS: "Показываю ваши заказы.",
            IntentType.SHOW_PROFILE: "Открываю профиль.",
            IntentType.SHOW_FAVORITES: "Показываю избранное.",
        }
        
        return intent_replies.get(intent, "Понял запрос, выполняю.")

    # =========================================================================
    # Загрузка конфигурации
    # =========================================================================

    def _build_slot_questions(self, config: Dict[str, Any]) -> Dict[IntentType, Dict[str, str]]:
        """Строит карту вопросов для слотов по интентам."""
        mapping: Dict[IntentType, Dict[str, str]] = {}
        
        for intent_name, intent_cfg in (config.get("intents") or {}).items():
            try:
                intent = IntentType(intent_name)
            except ValueError:
                continue
            
            questions = {str(k): str(v) for k, v in (intent_cfg.get("slot_questions") or {}).items()}
            if questions:
                mapping[intent] = questions
        
        return mapping

    def _load_dosage_forms(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Загружает формы выпуска из конфига."""
        entities = config.get("entities") or {}
        dosage_config = entities.get("dosage_forms") or {}
        
        if not dosage_config:
            return {
                "tablets": ["таблет", "табл."],
                "syrup": ["сироп"],
                "spray": ["спрей"],
                "capsules": ["капсул"],
                "drops": ["капли"],
            }
        
        return {
            str(form): [str(keyword).lower() for keyword in keywords or []]
            for form, keywords in dosage_config.items()
        }

    @staticmethod
    def _load_config(config_path: Path) -> Dict[str, Any]:
        """Загружает конфигурацию из YAML."""
        with config_path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}


# =============================================================================
# Синглтон
# =============================================================================

_slot_manager: SlotManager | None = None


def get_slot_manager() -> SlotManager:
    """Возвращает singleton-экземпляр SlotManager."""
    global _slot_manager
    if _slot_manager is None:
        _slot_manager = SlotManager()
    return _slot_manager


def reset_slot_manager() -> None:
    """Сбрасывает singleton (для тестов)."""
    global _slot_manager
    _slot_manager = None
