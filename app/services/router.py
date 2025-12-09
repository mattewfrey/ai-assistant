"""
RouterService - Детерминированный NLU-движок для классификации интентов.

================================================================================
АРХИТЕКТУРА
================================================================================

1. Предобработка текста (нормализация, морфология)
2. Многоуровневый матчинг:
   - Быстрый keyword-матчинг по триггерам
   - Регекс-матчинг для сложных паттернов
   - Эвристический детектор товарных запросов
3. Извлечение сущностей (слотов)
4. Расчет confidence и приоритизация
5. Расширенное логирование для debug

================================================================================
АНСАМБЛЬ С LLM
================================================================================

Router работает как первая линия классификации. Логика ансамбля:

1. **router_only** (confidence >= 0.85):
   - Router уверен в результате
   - LLM НЕ вызывается
   - Используем интент и слоты от Router'а
   
2. **router+slots** (confidence >= 0.85, но нужны слоты):
   - Router уверен в интенте
   - LLM может вызываться для дополнительного извлечения слотов
   
3. **router+llm** (0.5 <= confidence < 0.85):
   - Router не уверен, но есть кандидаты
   - LLM получает список кандидатов и выбирает
   - get_candidates() возвращает топ-3 интента
   
4. **llm_only** (confidence < 0.5 или нет матча):
   - Router не нашёл подходящего интента
   - LLM полностью определяет intent + slots

================================================================================
ПРИОРИТЕТЫ ИНТЕНТОВ
================================================================================

1. Корзина/Заказы (пользователь хочет действовать)
2. Профиль/Бонусы
3. Аптеки/Локации  
4. Правовая информация
5. Симптомы/Болезни
6. Поиск товаров
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from langsmith import traceable
import yaml

from ..intents import ActionChannel, IntentType
from ..models import ChatRequest, UserProfile
from .dialog_state import DialogState
from .slot_extraction import (
    SYMPTOM_PATTERN,
    SYMPTOM_STEMS,
    DISEASE_STEMS,
    extract_age,
    extract_price_max,
    extract_price_min,
    extract_dosage_form,
    extract_dosage_forms,
    extract_symptom,
    extract_symptoms,
    extract_disease,
    extract_diseases,
    extract_is_for_children,
    extract_special_filters,
    extract_all_entities,
    normalize_text,
    clean_symptom,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "router_config.yaml"
DEFAULT_PROMPT = "Уточните, пожалуйста."

# Минимальный confidence для уверенного матча (router_only)
MIN_CONFIDENT_MATCH = 0.85

# Порог для ансамбля router+llm (ниже этого - llm_only)
MIN_ENSEMBLE_THRESHOLD = 0.5

# Порог для срабатывания product-детектора
PRODUCT_DETECTOR_THRESHOLD = 0.65

# Максимальное количество кандидатов для LLM дизамбигуации
MAX_LLM_CANDIDATES = 3


@dataclass
class SlotDefinition:
    """Определение слота с промптом для уточнения."""
    name: str
    prompt: str


@dataclass
class RouterRule:
    """Правило маршрутизации для интента."""
    intent: IntentType
    channel: ActionChannel
    triggers: List[str] = field(default_factory=list)
    negative_triggers: List[str] = field(default_factory=list)
    slots_required: List[str] = field(default_factory=list)
    slot_questions: Dict[str, str] = field(default_factory=dict)
    priority: int = 50  # Приоритет (выше = важнее)


@dataclass
class RouterConfig:
    """Конфигурация роутера, загруженная из YAML."""
    rules: List[RouterRule]
    default_slot_questions: Dict[str, str]
    brand_keywords: List[str]
    symptom_keywords: List[str]
    dosage_forms: Dict[str, List[str]]
    synonyms: Dict[str, List[str]]


@dataclass
class MatchInfo:
    """Информация о матче для debug."""
    matched_triggers: List[str] = field(default_factory=list)
    matched_patterns: List[str] = field(default_factory=list)
    negative_hits: List[str] = field(default_factory=list)
    match_type: str = "none"  # keyword, regex, heuristic, symptom, disease, product
    raw_score: float = 0.0


@dataclass
class RouterResult:
    """Результат работы роутера."""
    matched: bool
    intent: Optional[IntentType] = None
    channel: Optional[ActionChannel] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[SlotDefinition] = field(default_factory=list)
    slot_questions: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    router_matched: bool = False
    
    # Расширенная диагностика
    match_info: Optional[MatchInfo] = None
    alternative_intents: List[Tuple[IntentType, float]] = field(default_factory=list)
    
    @property
    def extracted_slots(self) -> Dict[str, Any]:
        """Синоним для slots (обратная совместимость)."""
        return self.slots
    
    @property
    def is_confident(self) -> bool:
        """True если Router достаточно уверен (не нужен LLM)."""
        return self.matched and self.confidence >= MIN_CONFIDENT_MATCH
    
    @property
    def needs_llm_disambiguation(self) -> bool:
        """True если нужна помощь LLM для выбора между кандидатами."""
        return (
            self.matched 
            and MIN_ENSEMBLE_THRESHOLD <= self.confidence < MIN_CONFIDENT_MATCH
            and len(self.alternative_intents) > 0
        )
    
    @property
    def needs_full_llm(self) -> bool:
        """True если нужна полная классификация LLM."""
        return not self.matched or self.confidence < MIN_ENSEMBLE_THRESHOLD
    
    def get_pipeline_path(self) -> str:
        """
        Определяет путь пайплайна на основе confidence.
        
        Returns:
            - "router_only" - Router уверен
            - "router+slots" - Router уверен, но нужны слоты
            - "router+llm" - Router не уверен, нужна дизамбигуация
            - "llm_only" - Router не нашёл, полная классификация LLM
        """
        if not self.matched:
            return "llm_only"
        
        if self.confidence >= MIN_CONFIDENT_MATCH:
            if self.missing_slots:
                return "router+slots"
            return "router_only"
        
        if self.confidence >= MIN_ENSEMBLE_THRESHOLD:
            return "router+llm"
        
        return "llm_only"
    
    def get_candidates_for_llm(self) -> List[Tuple[str, float]]:
        """
        Возвращает кандидатов для LLM дизамбигуации.
        
        Returns:
            Список пар (intent_name, confidence) для топ кандидатов
        """
        candidates: List[Tuple[str, float]] = []
        
        # Добавляем основной интент
        if self.intent:
            candidates.append((self.intent.value, self.confidence))
        
        # Добавляем альтернативы
        for alt_intent, alt_conf in self.alternative_intents[:MAX_LLM_CANDIDATES - 1]:
            candidates.append((alt_intent.value, alt_conf))
        
        return candidates[:MAX_LLM_CANDIDATES]
    
    def to_debug_dict(self) -> Dict[str, Any]:
        """Формирует словарь для debug-вывода."""
        debug = {
            "router_matched": self.router_matched,
            "router_confidence": self.confidence,
            "intent": self.intent.value if self.intent else None,
            "channel": self.channel.value if self.channel else None,
            "extracted_slots": self.slots,
            "missing_slots": [s.name for s in self.missing_slots],
            "pipeline_path": self.get_pipeline_path(),
            "is_confident": self.is_confident,
            "needs_llm": not self.is_confident,
        }
        if self.match_info:
            debug["match_info"] = {
                "match_type": self.match_info.match_type,
                "matched_triggers": self.match_info.matched_triggers[:5],
                "matched_patterns": self.match_info.matched_patterns,
                "negative_hits": self.match_info.negative_hits,
                "raw_score": self.match_info.raw_score,
            }
        if self.alternative_intents:
            debug["alternative_intents"] = [
                {"intent": intent.value, "score": score}
                for intent, score in self.alternative_intents[:3]
            ]
            debug["llm_candidates"] = self.get_candidates_for_llm()
        return debug


class RouterService:
    """
    Config-driven роутер для детерминированной классификации интентов.
    
    Пайплайн:
    1. Нормализация входного текста
    2. Матчинг по правилам из router_config.yaml
    3. Эвристический детектор товарных/симптомных запросов
    4. Извлечение слотов
    5. Расчет confidence
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or CONFIG_PATH
        self._config = self._load_config(self._config_path)
        
        # Сортируем правила по приоритету (descending)
        self._rules = sorted(
            self._config.rules, 
            key=lambda r: r.priority, 
            reverse=True
        )
        
        self._brand_keywords = set(self._config.brand_keywords)
        self._symptom_keywords = set(self._config.symptom_keywords)
        self._dosage_form_keywords = self._config.dosage_forms
        self._synonyms = self._config.synonyms
        
        # Компилируем паттерн для симптомов
        self._symptom_regex = SYMPTOM_PATTERN
        
        # Кеш для быстрого поиска по триггерам
        self._trigger_to_rules: Dict[str, List[RouterRule]] = {}
        self._build_trigger_index()
        
        logger.info(
            "RouterService initialized with %d rules, %d brands, %d symptoms",
            len(self._rules),
            len(self._brand_keywords),
            len(self._symptom_keywords),
        )

    def _build_trigger_index(self) -> None:
        """Строит индекс триггер -> правила для быстрого поиска."""
        for rule in self._rules:
            for trigger in rule.triggers:
                if trigger not in self._trigger_to_rules:
                    self._trigger_to_rules[trigger] = []
                self._trigger_to_rules[trigger].append(rule)

    @traceable(run_type="chain", name="router_match")
    def match(
        self,
        *,
        request: ChatRequest,
        user_profile: UserProfile | None,
        dialog_state: DialogState | None,
        debug_builder: Any | None = None,
        trace_id: str | None = None,
    ) -> RouterResult:
        """
        Основной метод: пытается определить интент детерминированно.
        
        Args:
            request: Входящий запрос
            user_profile: Профиль пользователя (для дефолтов слотов)
            dialog_state: Текущее состояние диалога
            debug_builder: Билдер для debug-метаданных
            trace_id: ID трассировки
            
        Returns:
            RouterResult с результатом матчинга
        """
        message = (request.message or "").strip()
        if not message:
            return RouterResult(matched=False, router_matched=False)
        
        # Предобработка
        normalized = normalize_text(message)
        
        # Многоуровневый матчинг
        result = self._multi_level_match(
            message=message,
            normalized=normalized,
            user_profile=user_profile,
            dialog_state=dialog_state,
        )
        
        # Обновляем debug_builder
        if debug_builder and result.matched:
            debug_builder.set_router_matched(True)
            debug_builder.add_intent(result.intent.value if result.intent else None)
            if result.match_info:
                debug_builder.add_extra("router_match_type", result.match_info.match_type)
                debug_builder.add_extra("matched_triggers", result.match_info.matched_triggers[:5])
                debug_builder.add_extra("router_confidence", result.confidence)
        
        # Логируем результат
        if result.matched:
            logger.info(
                "trace_id=%s user_id=%s intent=%s confidence=%.2f match_type=%s slots=%s",
                trace_id or "-",
                getattr(request, "user_id", None) or "-",
                result.intent.value if result.intent else "-",
                result.confidence,
                result.match_info.match_type if result.match_info else "unknown",
                list(result.slots.keys()),
            )
        else:
            logger.debug(
                "trace_id=%s user_id=%s Router no match for: %s",
                trace_id or "-",
                getattr(request, "user_id", None) or "-",
                message[:50],
            )
        
        return result

    def _multi_level_match(
        self,
        message: str,
        normalized: str,
        user_profile: UserProfile | None,
        dialog_state: DialogState | None,
    ) -> RouterResult:
        """
        Многоуровневый матчинг с приоритизацией.
        
        Уровни:
        1. Точный keyword-матчинг по правилам (с учетом negative triggers)
        2. Детектор симптомных/болезневых запросов
        3. Эвристический детектор товарных запросов
        """
        candidates: List[Tuple[RouterResult, int]] = []  # (result, priority)
        
        # Уровень 1: Keyword матчинг по правилам
        for rule in self._rules:
            match_result = self._try_match_rule(rule, message, normalized, user_profile, dialog_state)
            if match_result and match_result.matched:
                candidates.append((match_result, rule.priority))
        
        # Уровень 2: Детектор болезней (сначала болезни, т.к. они более специфичны)
        if not candidates:
            disease_result = self._detect_disease_query(message, normalized, user_profile, dialog_state)
            if disease_result and disease_result.matched:
                candidates.append((disease_result, 42))  # Приоритет как у FIND_BY_DISEASE
        
        # Уровень 3: Симптомный детектор (только если болезни не найдены)
        if not candidates:
            symptom_result = self._detect_symptom_query(message, normalized, user_profile, dialog_state)
            if symptom_result and symptom_result.matched:
                candidates.append((symptom_result, 38))  # Приоритет как у FIND_BY_SYMPTOM
        
        # Уровень 4: Детектор товарных запросов (если все еще пусто)
        if not candidates:
            product_result = self._detect_product_query(message, normalized, user_profile, dialog_state)
            if product_result and product_result.matched:
                candidates.append((product_result, 35))  # Приоритет как у FIND_PRODUCT_BY_NAME
        
        # Выбираем лучший результат
        if not candidates:
            return RouterResult(matched=False, router_matched=False)
        
        # Сортируем по: приоритет (desc), confidence (desc)
        candidates.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)
        
        best_result = candidates[0][0]
        
        # Добавляем альтернативные интенты
        if len(candidates) > 1:
            best_result.alternative_intents = [
                (c[0].intent, c[0].confidence) 
                for c in candidates[1:4] 
                if c[0].intent
            ]
        
        return best_result

    def _try_match_rule(
        self,
        rule: RouterRule,
        message: str,
        normalized: str,
        user_profile: UserProfile | None,
        dialog_state: DialogState | None,
    ) -> RouterResult | None:
        """Пытается применить конкретное правило."""
        match_info = MatchInfo()
        
        # Проверяем negative triggers сначала
        for neg_trigger in rule.negative_triggers:
            if neg_trigger in normalized:
                match_info.negative_hits.append(neg_trigger)
                return None  # Negative trigger сработал - отклоняем
        
        # Проверяем positive triggers
        matched_triggers = []
        for trigger in rule.triggers:
            if trigger in normalized:
                matched_triggers.append(trigger)
        
        if not matched_triggers:
            # Специальная обработка для FIND_BY_SYMPTOM
            if rule.intent == IntentType.FIND_BY_SYMPTOM:
                if self._matches_symptom_keywords(normalized):
                    match_info.match_type = "symptom_keyword"
                    match_info.matched_patterns.append("symptom_stem")
                elif self._symptom_regex.search(message):
                    # Дополнительная проверка: убедимся что найденный "симптом" 
                    # действительно содержит медицинские термины
                    regex_match = self._symptom_regex.search(message)
                    if regex_match:
                        candidate_symptom = regex_match.group("symptom").lower().strip()
                        # Проверяем что кандидат содержит хотя бы один медицинский термин
                        has_medical_term = any(
                            stem in candidate_symptom 
                            for stem in SYMPTOM_STEMS.keys()
                        )
                        if has_medical_term:
                            match_info.match_type = "symptom_regex"
                            match_info.matched_patterns.append("symptom_pattern")
                        else:
                            # Regex сработал, но это не медицинский симптом — пропускаем
                            return None
                    else:
                        return None
                else:
                    return None
            # Специальная обработка для FIND_BY_DISEASE
            elif rule.intent == IntentType.FIND_BY_DISEASE:
                if self._matches_disease_keywords(normalized):
                    match_info.match_type = "disease_keyword"
                    match_info.matched_patterns.append("disease_stem")
                else:
                    return None
            else:
                return None  # Нет совпадений
        else:
            match_info.match_type = "keyword"
            match_info.matched_triggers = matched_triggers
        
        # Успешный матч - строим результат
        return self._build_rule_result(
            rule=rule,
            message=message,
            normalized_message=normalized,
            user_profile=user_profile,
            dialog_state=dialog_state,
            match_info=match_info,
        )

    def _build_rule_result(
        self,
        *,
        rule: RouterRule,
        message: str,
        normalized_message: str,
        user_profile: UserProfile | None,
        dialog_state: DialogState | None,
        match_info: MatchInfo,
    ) -> RouterResult:
        """Строит RouterResult из сработавшего правила."""
        # Извлекаем слоты
        slots = self._extract_slots(rule.intent, message, normalized_message, user_profile, dialog_state)
        
        # Определяем недостающие слоты
        required_slots = list(rule.slots_required)
        
        # Оптимизация: если форма выпуска уже указана для FIND_BY_SYMPTOM, не требуем возраст
        if rule.intent == IntentType.FIND_BY_SYMPTOM and slots.get("dosage_form"):
            required_slots = [slot for slot in required_slots if slot != "age"]
        
        missing_slots = self._get_missing_slots(rule, slots, required_slots)
        
        # Рассчитываем confidence
        confidence = self._calculate_confidence(
            match_info=match_info,
            slots=slots,
            missing_slots=missing_slots,
            rule=rule,
        )
        
        match_info.raw_score = confidence
        
        return RouterResult(
            matched=True,
            intent=rule.intent,
            channel=rule.channel,
            slots=slots,
            missing_slots=missing_slots,
            slot_questions=self._get_questions_map(rule),
            confidence=confidence,
            router_matched=True,
            match_info=match_info,
        )

    def _detect_symptom_query(
        self,
        message: str,
        normalized: str,
        user_profile: UserProfile | None,
        dialog_state: DialogState | None,
    ) -> RouterResult | None:
        """
        Детектор симптомных запросов на основе словаря симптомов.
        """
        # Извлекаем симптомы
        symptoms = extract_symptoms(message)
        if not symptoms:
            return None
        
        match_info = MatchInfo(
            match_type="symptom_detection",
            matched_patterns=symptoms[:3],
        )
        
        slots = self._extract_slots(IntentType.FIND_BY_SYMPTOM, message, normalized, user_profile, dialog_state)
        slots["symptom"] = symptoms[0]
        if len(symptoms) > 1:
            slots["symptoms"] = symptoms
        
        # Ищем правило FIND_BY_SYMPTOM для получения slot_questions
        symptom_rule = next(
            (r for r in self._rules if r.intent == IntentType.FIND_BY_SYMPTOM),
            None
        )
        
        required_slots = symptom_rule.slots_required if symptom_rule else ["symptom"]
        missing_slots = []
        
        if symptom_rule:
            missing_slots = self._get_missing_slots(symptom_rule, slots, required_slots)
        
        confidence = 0.85 if symptoms else 0.7
        if slots.get("age"):
            confidence = min(confidence + 0.1, 0.98)
        
        return RouterResult(
            matched=True,
            intent=IntentType.FIND_BY_SYMPTOM,
            channel=ActionChannel.DATA,
            slots=slots,
            missing_slots=missing_slots,
            slot_questions=symptom_rule.slot_questions if symptom_rule else {},
            confidence=confidence,
            router_matched=True,
            match_info=match_info,
        )

    def _detect_disease_query(
        self,
        message: str,
        normalized: str,
        user_profile: UserProfile | None,
        dialog_state: DialogState | None,
    ) -> RouterResult | None:
        """
        Детектор запросов по заболеваниям на основе словаря болезней.
        """
        # Извлекаем заболевания
        diseases = extract_diseases(message)
        if not diseases:
            return None
        
        match_info = MatchInfo(
            match_type="disease_detection",
            matched_patterns=diseases[:3],
        )
        
        slots = self._extract_slots(IntentType.FIND_BY_DISEASE, message, normalized, user_profile, dialog_state)
        slots["disease"] = diseases[0]
        if len(diseases) > 1:
            slots["diseases"] = diseases
        
        # Ищем правило FIND_BY_DISEASE для получения slot_questions
        disease_rule = next(
            (r for r in self._rules if r.intent == IntentType.FIND_BY_DISEASE),
            None
        )
        
        required_slots = disease_rule.slots_required if disease_rule else ["disease"]
        missing_slots = []
        
        if disease_rule:
            missing_slots = self._get_missing_slots(disease_rule, slots, required_slots)
        
        confidence = 0.88 if diseases else 0.75
        if slots.get("age"):
            confidence = min(confidence + 0.1, 0.98)
        
        return RouterResult(
            matched=True,
            intent=IntentType.FIND_BY_DISEASE,
            channel=ActionChannel.DATA,
            slots=slots,
            missing_slots=missing_slots,
            slot_questions=disease_rule.slot_questions if disease_rule else {},
            confidence=confidence,
            router_matched=True,
            match_info=match_info,
        )

    def _detect_product_query(
        self,
        message: str,
        normalized: str,
        user_profile: UserProfile | None,
        dialog_state: DialogState | None,
    ) -> RouterResult | None:
        """
        Эвристический детектор товарных запросов.
        
        Признаки товарного запроса:
        - Короткий запрос (1-6 слов)
        - Содержит цифры (дозировка: 500мг, 100мл)
        - Начинается с заглавной (бренд)
        - Содержит латиницу (международное название)
        - Совпадает с известным брендом
        """
        # Слишком длинные запросы отклоняем
        if len(message) > 96:
            return None
        
        # Проверяем известные бренды
        brand_hit = any(brand in normalized for brand in self._brand_keywords)
        
        # Эвристики для короткого запроса
        looks_like_product = brand_hit or self._looks_like_product_query(message)
        
        if not looks_like_product:
            return None
        
        match_info = MatchInfo(
            match_type="product_heuristic",
            matched_patterns=["brand"] if brand_hit else ["short_query"],
        )
        
        # Извлекаем слоты
        slots = {
            "name": message.strip(),
            "product_name": message.strip(),
        }
        slots.update(self._extract_slots(IntentType.FIND_PRODUCT_BY_NAME, message, normalized, user_profile, dialog_state))
        
        # Рассчитываем confidence
        confidence = 0.75 if brand_hit else 0.65
        
        # Бустим confidence если есть дозировка
        if re.search(r"\d+\s*(?:мг|мл|г|mg|ml|g)", normalized):
            confidence = min(confidence + 0.1, 0.9)
            match_info.matched_patterns.append("dosage")
        
        return RouterResult(
            matched=True,
            intent=IntentType.FIND_PRODUCT_BY_NAME,
            channel=ActionChannel.DATA,
            slots=slots,
            slot_questions={},
            confidence=confidence,
            router_matched=True,
            match_info=match_info,
        )

    def _extract_slots(
        self,
        intent: IntentType | None,
        message: str,
        normalized: str,
        user_profile: UserProfile | None,
        dialog_state: DialogState | None,
    ) -> Dict[str, Any]:
        """
        Комплексное извлечение слотов из сообщения.
        
        Источники (в порядке приоритета):
        1. Текущее сообщение (самый высокий приоритет)
        2. Состояние диалога
        3. Профиль пользователя (дефолты)
        """
        # Используем продвинутое извлечение
        extraction = extract_all_entities(message, self._dosage_form_keywords)
        slots = extraction.to_slots_dict()
        
        # Для симптомных интентов извлекаем симптом
        if intent in (IntentType.FIND_BY_SYMPTOM, IntentType.SYMPTOM_TO_PRODUCT):
            if not slots.get("symptom"):
                symptom = self._extract_symptom_phrase(message, normalized)
                if symptom:
                    slots["symptom"] = symptom
        
        # Для болезневых интентов
        if intent in (IntentType.FIND_BY_DISEASE, IntentType.DISEASE_TO_PRODUCT):
            if not slots.get("disease"):
                disease = extract_disease(message)
                if disease:
                    slots["disease"] = disease
        
        # Применяем дефолты из профиля
        if user_profile:
            slots = self._apply_profile_defaults(slots, user_profile)
        
        # Мержим с состоянием диалога (не перезаписываем свежие данные)
        if dialog_state and dialog_state.slots:
            for key, value in dialog_state.slots.items():
                if key not in slots or slots[key] is None:
                    slots[key] = value
        
        # Фильтруем None значения
        return {k: v for k, v in slots.items() if v is not None}

    def _apply_profile_defaults(self, slots: Dict[str, Any], user_profile: UserProfile) -> Dict[str, Any]:
        """Применяет дефолты из профиля пользователя."""
        prefs = user_profile.preferences
        
        if not slots.get("age") and getattr(prefs, "age", None):
            slots["age"] = prefs.age
        
        if not slots.get("price_max"):
            default_price = getattr(prefs, "default_max_price", None)
            if default_price is not None:
                slots["price_max"] = default_price
        
        preferred_forms = getattr(prefs, "preferred_forms", None)
        if preferred_forms:
            if not slots.get("preferred_forms"):
                slots["preferred_forms"] = preferred_forms
            if not slots.get("dosage_form"):
                slots["dosage_form"] = preferred_forms[0]
        
        # Копируем булевые флаги
        if getattr(prefs, "sugar_free", False):
            slots.setdefault("sugar_free", True)
        if getattr(prefs, "for_children", False):
            slots.setdefault("is_for_children", True)
        
        return slots

    def _get_missing_slots(
        self, 
        rule: RouterRule, 
        slots: Dict[str, Any], 
        required_slots: List[str] | None = None
    ) -> List[SlotDefinition]:
        """Определяет недостающие обязательные слоты."""
        missing: List[SlotDefinition] = []
        required = required_slots if required_slots is not None else rule.slots_required
        
        for slot_name in required:
            value = slots.get(slot_name)
            if value is None or value == "" or value == []:
                prompt = self._get_questions_map(rule).get(slot_name, DEFAULT_PROMPT)
                missing.append(SlotDefinition(name=slot_name, prompt=prompt))
        
        return missing

    def _get_questions_map(self, rule: RouterRule) -> Dict[str, str]:
        """Возвращает объединенную карту вопросов для слотов."""
        result = dict(self._config.default_slot_questions)
        result.update(rule.slot_questions or {})
        return result

    def _calculate_confidence(
        self,
        match_info: MatchInfo,
        slots: Dict[str, Any],
        missing_slots: List[SlotDefinition],
        rule: RouterRule,
    ) -> float:
        """
        Рассчитывает confidence матча.
        
        Факторы:
        - Тип матча (keyword > symptom > heuristic)
        - Количество совпавших триггеров
        - Наличие/отсутствие слотов
        - Приоритет правила
        """
        base_confidence = 0.7
        
        # Буст за тип матча
        if match_info.match_type == "keyword":
            base_confidence = 0.85
            # Буст за количество триггеров
            trigger_count = len(match_info.matched_triggers)
            if trigger_count >= 2:
                base_confidence = min(base_confidence + 0.05 * trigger_count, 0.95)
        elif match_info.match_type in ("symptom_keyword", "symptom_detection"):
            base_confidence = 0.80
        elif match_info.match_type == "disease_keyword":
            base_confidence = 0.82
        elif match_info.match_type == "product_heuristic":
            base_confidence = 0.70
        
        # Штраф за недостающие слоты
        if missing_slots:
            penalty = 0.05 * len(missing_slots)
            base_confidence = max(base_confidence - penalty, 0.5)
        
        # Буст за высокий приоритет правила
        if rule.priority >= 90:
            base_confidence = min(base_confidence + 0.05, 0.98)
        
        return round(base_confidence, 2)

    def _extract_symptom_phrase(self, message: str, normalized: str) -> Optional[str]:
        """Извлекает фразу симптома из сообщения."""
        # Используем продвинутое извлечение
        symptom = extract_symptom(message)
        if symptom:
            return symptom
        
        # Fallback на regex
        match = self._symptom_regex.search(message)
        if match:
            raw = match.group("symptom")
            cleaned = clean_symptom(raw)
            return cleaned or raw.strip()
        
        # Fallback на ключевые слова
        for keyword in self._symptom_keywords:
            if keyword in normalized:
                return keyword
        
        return None

    def _matches_symptom_keywords(self, normalized: str) -> bool:
        """Проверяет наличие ключевых слов симптомов."""
        # Проверяем собственные ключевые слова
        if any(keyword in normalized for keyword in self._symptom_keywords):
            return True
        
        # Проверяем словарь из slot_extraction
        for stem in SYMPTOM_STEMS:
            if stem in normalized:
                return True
        
        return False

    def _matches_disease_keywords(self, normalized: str) -> bool:
        """Проверяет наличие ключевых слов заболеваний."""
        for stem in DISEASE_STEMS:
            if stem in normalized:
                return True
        return False

    def _looks_like_product_query(self, message: str) -> bool:
        """Эвристика: похоже ли сообщение на товарный запрос."""
        tokens = message.split()
        if not (1 <= len(tokens) <= 6):
            return False
        
        # Содержит цифры (дозировка)
        has_digit = any(char.isdigit() for char in message)
        
        # Начинается с заглавной (бренд)
        has_initial_upper = message[0].isupper() if message else False
        
        # Содержит латиницу (МНН)
        has_latin = any(char.isascii() and char.isalpha() for char in message)
        
        return has_digit or has_initial_upper or has_latin

    # =========================================================================
    # Загрузка конфигурации
    # =========================================================================

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _load_config(config_path: Path) -> RouterConfig:
        """Загружает и парсит конфигурацию из YAML."""
        if not config_path.exists():
            raise FileNotFoundError(f"Router config not found at {config_path}")
        
        with config_path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        
        # Парсим defaults
        defaults = (data.get("defaults") or {}).get("slot_questions") or {}
        
        # Парсим правила интентов
        intents = data.get("intents") or {}
        rules: List[RouterRule] = []
        
        for intent_name, config in intents.items():
            try:
                intent = IntentType(intent_name)
            except ValueError:
                logger.warning("Unknown intent in config: %s", intent_name)
                continue
            
            channel_value = config.get("channel", "other")
            try:
                channel = ActionChannel(channel_value)
            except ValueError:
                channel = ActionChannel.OTHER
            
            triggers = [str(trigger).lower() for trigger in (config.get("triggers") or [])]
            negative_triggers = [str(t).lower() for t in (config.get("negative_triggers") or [])]
            slots_required = [str(slot) for slot in (config.get("slots_required") or [])]
            slot_questions = {str(k): str(v) for k, v in (config.get("slot_questions") or {}).items()}
            priority = int(config.get("priority", 50))
            
            rules.append(RouterRule(
                intent=intent,
                channel=channel,
                triggers=triggers,
                negative_triggers=negative_triggers,
                slots_required=slots_required,
                slot_questions=slot_questions,
                priority=priority,
            ))
        
        # Парсим entities
        entities = data.get("entities") or {}
        brand_keywords = [str(value).lower() for value in entities.get("product_brands") or []]
        symptom_keywords = [str(value).lower() for value in entities.get("symptom_keywords") or []]
        
        # Парсим формы выпуска
        dosage_forms_raw = entities.get("dosage_forms") or {}
        dosage_forms: Dict[str, List[str]] = {}
        for form, keywords in dosage_forms_raw.items():
            dosage_forms[str(form)] = [str(keyword).lower() for keyword in keywords or []]
        
        # Парсим синонимы
        synonyms_raw = entities.get("synonyms") or {}
        synonyms: Dict[str, List[str]] = {}
        for key, values in synonyms_raw.items():
            synonyms[str(key)] = [str(v).lower() for v in values or []]
        
        return RouterConfig(
            rules=rules,
            default_slot_questions={str(k): str(v) for k, v in defaults.items()},
            brand_keywords=brand_keywords,
            symptom_keywords=symptom_keywords,
            dosage_forms=dosage_forms,
            synonyms=synonyms,
        )


# =============================================================================
# Синглтон
# =============================================================================

_router_service: RouterService | None = None


def get_router_service() -> RouterService:
    """Возвращает singleton-экземпляр RouterService."""
    global _router_service
    if _router_service is None:
        _router_service = RouterService()
    return _router_service


def reset_router_service() -> None:
    """Сбрасывает singleton (для тестов)."""
    global _router_service
    _router_service = None
