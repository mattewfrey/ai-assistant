"""
DebugMetaBuilder - Построитель диагностической информации для meta.debug.

================================================================================
СОБИРАЕМАЯ ИНФОРМАЦИЯ
================================================================================

Основные флаги:
- llm_used: bool - использовался ли LLM
- llm_cached: bool - был ли ответ из кэша
- router_matched: bool - сработал ли Router
- slot_filling_used: bool - использовалось ли уточнение слотов

Pipeline информация:
- pipeline_path: str - путь обработки:
  * "router_only" - только Router
  * "router+slots" - Router + уточнение слотов
  * "router+llm" - Router + LLM дизамбигуация
  * "llm_only" - только LLM
  * "llm+platform" - LLM + вызов платформы
- intent_chain: List[str] - цепочка интентов

Router информация:
- router_confidence: float - уверенность Router'а
- router_match_type: str - тип матча (keyword, symptom, etc.)
- matched_triggers: List[str] - сработавшие триггеры
- router_candidates: List[dict] - кандидаты для LLM

LLM информация:
- llm_confidence: float - уверенность LLM
- llm_backend: str - бэкенд (langchain)
- llm_reasoning: str - объяснение от LLM

Слоты/сущности:
- extracted_entities_before: Dict - сущности до LLM
- extracted_entities_after: Dict - сущности после LLM
- filled_slots: List[str] - заполненные слоты
- missing_slots: List[str] - недостающие слоты
- pending_slots: bool - есть ли ожидающие слоты

Идентификаторы:
- trace_id: str - ID трассировки
- request_id: str - ID запроса
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DebugMetaBuilder:
    """
    Билдер для формирования meta.debug payload.
    
    Использование:
        builder = DebugMetaBuilder(request_id="abc123")
        builder.set_router_matched(True)
        builder.set_llm_used(False)
        builder.add_intent("FIND_BY_SYMPTOM")
        debug = builder.build()
    """

    def __init__(
        self,
        *,
        trace_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        # Основные флаги
        self._llm_used: Optional[bool] = None
        self._llm_cached: Optional[bool] = None
        self._router_matched: Optional[bool] = None
        self._slot_filling_used: Optional[bool] = None
        
        # Цепочка интентов
        self._intent_chain: List[str] = []
        
        # Источник/путь обработки
        self._source: Optional[str] = None
        self._pipeline_path: Optional[str] = None
        
        # Идентификаторы
        self._trace_id = trace_id
        self._request_id = request_id
        
        # Слоты
        self._pending_slots: Optional[bool] = None
        self._filled_slots: List[str] = []
        self._missing_slots: List[str] = []
        
        # Router-специфичная информация
        self._router_confidence: Optional[float] = None
        self._router_match_type: Optional[str] = None
        self._matched_triggers: List[str] = []
        self._router_candidates: List[Dict[str, Any]] = []  # Кандидаты для LLM
        
        # LLM-специфичная информация
        self._llm_confidence: Optional[float] = None
        self._llm_backend: Optional[str] = None
        self._llm_reasoning: Optional[str] = None  # Объяснение от LLM
        
        # Извлеченные сущности (до и после LLM)
        self._extracted_entities: Dict[str, Any] = {}
        self._extracted_entities_before: Dict[str, Any] = {}  # До LLM
        self._extracted_entities_after: Dict[str, Any] = {}  # После LLM
        
        # Дополнительные данные
        self._extra: Dict[str, Any] = {}

    # =========================================================================
    # Основные сеттеры (возвращают self для chaining)
    # =========================================================================

    def set_llm_used(self, used: bool, *, cached: bool | None = None) -> "DebugMetaBuilder":
        """Устанавливает флаг использования LLM."""
        self._llm_used = used
        if cached is not None:
            self._llm_cached = cached
        return self

    def set_llm_cached(self, cached: bool) -> "DebugMetaBuilder":
        """Устанавливает флаг кэширования LLM-ответа."""
        self._llm_cached = cached
        return self

    def set_router_matched(self, matched: bool) -> "DebugMetaBuilder":
        """Устанавливает флаг срабатывания роутера."""
        self._router_matched = matched
        return self

    def set_slot_filling_used(self, used: bool) -> "DebugMetaBuilder":
        """Устанавливает флаг использования slot filling."""
        self._slot_filling_used = used
        return self

    def add_intent(self, intent: str | None) -> "DebugMetaBuilder":
        """Добавляет интент в цепочку."""
        if intent and intent not in self._intent_chain:
            self._intent_chain.append(intent)
        return self

    def set_source(self, source: str) -> "DebugMetaBuilder":
        """Устанавливает источник ответа."""
        self._source = source
        return self

    def set_pipeline_path(self, path: str) -> "DebugMetaBuilder":
        """
        Устанавливает путь пайплайна.
        
        Возможные значения:
        - "router_only" - только роутер
        - "router+slots" - роутер с уточнением слотов  
        - "router+llm" - роутер + LLM для усиления
        - "llm_only" - только LLM
        - "llm+platform" - LLM + вызов платформы
        - "local_router" - локальный быстрый роутер
        """
        self._pipeline_path = path
        return self

    def set_trace_id(self, trace_id: str | None) -> "DebugMetaBuilder":
        """Устанавливает trace_id."""
        if trace_id:
            self._trace_id = trace_id
        return self

    def set_request_id(self, request_id: str | None) -> "DebugMetaBuilder":
        """Устанавливает request_id."""
        if request_id:
            self._request_id = request_id
        return self

    def set_pending_slots(self, pending: bool) -> "DebugMetaBuilder":
        """Устанавливает флаг ожидающих слотов."""
        self._pending_slots = pending
        return self

    # =========================================================================
    # Router-специфичные сеттеры
    # =========================================================================

    def set_router_confidence(self, confidence: float) -> "DebugMetaBuilder":
        """Устанавливает confidence роутера."""
        self._router_confidence = confidence
        return self

    def set_router_match_type(self, match_type: str) -> "DebugMetaBuilder":
        """
        Устанавливает тип матча роутера.
        
        Возможные значения:
        - "keyword" - по ключевым словам
        - "symptom_keyword" - по симптому
        - "symptom_regex" - по regex симптома
        - "disease_keyword" - по заболеванию
        - "product_heuristic" - эвристика товара
        - "symptom_detection" - детектор симптомов
        """
        self._router_match_type = match_type
        return self

    def set_matched_triggers(self, triggers: List[str]) -> "DebugMetaBuilder":
        """Устанавливает сработавшие триггеры."""
        self._matched_triggers = triggers[:10]  # Ограничиваем
        return self

    # =========================================================================
    # LLM-специфичные сеттеры
    # =========================================================================

    def set_llm_confidence(self, confidence: float) -> "DebugMetaBuilder":
        """Устанавливает confidence LLM."""
        self._llm_confidence = confidence
        return self

    def set_llm_backend(self, backend: str) -> "DebugMetaBuilder":
        """Устанавливает бэкенд LLM (langchain, openai, etc)."""
        self._llm_backend = backend
        return self

    def set_llm_reasoning(self, reasoning: str) -> "DebugMetaBuilder":
        """Устанавливает объяснение от LLM."""
        self._llm_reasoning = reasoning
        return self

    def set_router_candidates(self, candidates: List[Dict[str, Any]]) -> "DebugMetaBuilder":
        """Устанавливает кандидатов Router'а для LLM дизамбигуации."""
        self._router_candidates = candidates[:5]
        return self

    # =========================================================================
    # Слоты и сущности
    # =========================================================================

    def set_filled_slots(self, slots: List[str]) -> "DebugMetaBuilder":
        """Устанавливает список заполненных слотов."""
        self._filled_slots = slots
        return self

    def set_missing_slots(self, slots: List[str]) -> "DebugMetaBuilder":
        """Устанавливает список недостающих слотов."""
        self._missing_slots = slots
        return self

    def set_extracted_entities(self, entities: Dict[str, Any]) -> "DebugMetaBuilder":
        """Устанавливает извлеченные сущности."""
        self._extracted_entities = entities
        return self

    def add_extracted_entity(self, name: str, value: Any) -> "DebugMetaBuilder":
        """Добавляет извлеченную сущность."""
        self._extracted_entities[name] = value
        return self

    def set_extracted_entities_before(self, entities: Dict[str, Any]) -> "DebugMetaBuilder":
        """Устанавливает сущности ДО обработки LLM (от Router'а)."""
        self._extracted_entities_before = entities
        return self

    def set_extracted_entities_after(self, entities: Dict[str, Any]) -> "DebugMetaBuilder":
        """Устанавливает сущности ПОСЛЕ обработки LLM."""
        self._extracted_entities_after = entities
        return self

    # =========================================================================
    # Дополнительные данные
    # =========================================================================

    def add_extra(self, key: str, value: Any) -> "DebugMetaBuilder":
        """Добавляет произвольные данные."""
        self._extra[key] = value
        return self

    def merge_existing(self, debug: Dict[str, Any] | None) -> "DebugMetaBuilder":
        """
        Мержит существующий debug payload.
        
        Используется для сохранения данных из предыдущих этапов пайплайна.
        """
        if not debug:
            return self
        
        # Основные флаги
        if debug.get("llm_used") is not None:
            self.set_llm_used(debug.get("llm_used"))
        if "llm_cached" in debug:
            self.set_llm_cached(bool(debug.get("llm_cached")))
        if "router_matched" in debug:
            self.set_router_matched(bool(debug.get("router_matched")))
        if "slot_filling_used" in debug:
            self.set_slot_filling_used(bool(debug.get("slot_filling_used")))
        
        # Цепочка интентов
        if debug.get("intent_chain"):
            for intent in debug.get("intent_chain") or []:
                self.add_intent(intent)
        
        # Источник
        if debug.get("source"):
            self.set_source(str(debug.get("source")))
        if debug.get("pipeline_path"):
            self.set_pipeline_path(str(debug.get("pipeline_path")))
        
        # Идентификаторы
        if debug.get("trace_id"):
            self.set_trace_id(str(debug.get("trace_id")))
        if debug.get("request_id"):
            self.set_request_id(str(debug.get("request_id")))
        
        # Слоты
        if "pending_slots" in debug:
            self.set_pending_slots(bool(debug.get("pending_slots")))
        if debug.get("filled_slots"):
            self.set_filled_slots(debug.get("filled_slots"))
        if debug.get("missing_slots"):
            self.set_missing_slots(debug.get("missing_slots"))
        
        # Router
        if debug.get("router_confidence") is not None:
            self.set_router_confidence(float(debug.get("router_confidence")))
        if debug.get("router_match_type"):
            self.set_router_match_type(str(debug.get("router_match_type")))
        if debug.get("matched_triggers"):
            self.set_matched_triggers(debug.get("matched_triggers"))
        
        # LLM
        if debug.get("llm_confidence") is not None:
            self.set_llm_confidence(float(debug.get("llm_confidence")))
        if debug.get("llm_backend"):
            self.set_llm_backend(str(debug.get("llm_backend")))
        
        # Сущности
        if debug.get("extracted_entities"):
            self._extracted_entities.update(debug.get("extracted_entities") or {})
        
        # Все остальное в extra
        known_keys = {
            "llm_used", "llm_cached", "router_matched", "slot_filling_used",
            "intent_chain", "source", "pipeline_path", "trace_id", "request_id",
            "pending_slots", "filled_slots", "missing_slots",
            "router_confidence", "router_match_type", "matched_triggers",
            "llm_confidence", "llm_backend", "extracted_entities",
        }
        for key, value in debug.items():
            if key not in known_keys:
                self._extra[key] = value
        
        return self

    # =========================================================================
    # Построение результата
    # =========================================================================

    def build(self) -> Dict[str, Any]:
        """
        Строит финальный debug payload.
        
        Returns:
            Словарь с диагностической информацией
        """
        # Основные флаги
        payload: Dict[str, Any] = {
            "llm_used": bool(self._llm_used) if self._llm_used is not None else False,
            "llm_cached": bool(self._llm_cached) if self._llm_cached is not None else False,
            "router_matched": bool(self._router_matched) if self._router_matched is not None else False,
            "slot_filling_used": bool(self._slot_filling_used) if self._slot_filling_used is not None else False,
            "intent_chain": list(self._intent_chain),
        }
        
        # Источник и путь
        if self._source:
            payload["source"] = self._source
        else:
            payload["source"] = self._infer_source()
        
        if self._pipeline_path:
            payload["pipeline_path"] = self._pipeline_path
        else:
            payload["pipeline_path"] = self._infer_pipeline_path()
        
        # Идентификаторы
        if self._trace_id:
            payload["trace_id"] = self._trace_id
        if self._request_id:
            payload["request_id"] = self._request_id
        
        # Слоты
        if self._pending_slots is not None:
            payload["pending_slots"] = self._pending_slots
        if self._filled_slots:
            payload["filled_slots"] = self._filled_slots
        if self._missing_slots:
            payload["missing_slots"] = self._missing_slots
        
        # Router-информация
        if self._router_confidence is not None:
            payload["router_confidence"] = round(self._router_confidence, 2)
        if self._router_match_type:
            payload["router_match_type"] = self._router_match_type
        if self._matched_triggers:
            payload["matched_triggers"] = self._matched_triggers
        if self._router_candidates:
            payload["router_candidates"] = self._router_candidates
        
        # LLM-информация
        if self._llm_confidence is not None:
            payload["llm_confidence"] = round(self._llm_confidence, 2)
        if self._llm_backend:
            payload["llm_backend"] = self._llm_backend
        if self._llm_reasoning:
            payload["llm_reasoning"] = self._llm_reasoning
        
        # Сущности (включая до/после LLM)
        if self._extracted_entities:
            payload["extracted_entities"] = self._extracted_entities
        if self._extracted_entities_before:
            payload["extracted_entities_before"] = self._extracted_entities_before
        if self._extracted_entities_after:
            payload["extracted_entities_after"] = self._extracted_entities_after
        
        # Дополнительные данные
        payload.update(self._extra)
        
        return payload

    def _infer_source(self) -> str:
        """Автоматически определяет source на основе флагов."""
        if self._router_matched:
            if self._slot_filling_used:
                return "router+slots"
            return "router"
        
        if self._slot_filling_used:
            return "slots"
        
        if self._llm_used:
            return "llm"
        
        return "unknown"

    def _infer_pipeline_path(self) -> str:
        """Автоматически определяет pipeline_path на основе флагов."""
        if self._source == "local_router":
            return "local_router"
        
        if self._router_matched and self._llm_used:
            return "router+llm"
        
        if self._router_matched and self._slot_filling_used:
            return "router+slots"
        
        if self._router_matched:
            return "router_only"
        
        if self._llm_used:
            # Проверяем, был ли вызов платформы (по наличию данных)
            has_platform_data = self._extra.get("has_platform_data", False)
            if has_platform_data:
                return "llm+platform"
            return "llm_only"
        
        return "unknown"

    # =========================================================================
    # Утилиты
    # =========================================================================

    def copy(self) -> "DebugMetaBuilder":
        """Создает копию билдера."""
        new_builder = DebugMetaBuilder(
            trace_id=self._trace_id,
            request_id=self._request_id,
        )
        new_builder._llm_used = self._llm_used
        new_builder._llm_cached = self._llm_cached
        new_builder._router_matched = self._router_matched
        new_builder._slot_filling_used = self._slot_filling_used
        new_builder._intent_chain = list(self._intent_chain)
        new_builder._source = self._source
        new_builder._pipeline_path = self._pipeline_path
        new_builder._pending_slots = self._pending_slots
        new_builder._filled_slots = list(self._filled_slots)
        new_builder._missing_slots = list(self._missing_slots)
        new_builder._router_confidence = self._router_confidence
        new_builder._router_match_type = self._router_match_type
        new_builder._matched_triggers = list(self._matched_triggers)
        new_builder._llm_confidence = self._llm_confidence
        new_builder._llm_backend = self._llm_backend
        new_builder._extracted_entities = dict(self._extracted_entities)
        new_builder._extra = dict(self._extra)
        return new_builder

    def __repr__(self) -> str:
        return (
            f"DebugMetaBuilder("
            f"router={self._router_matched}, "
            f"llm={self._llm_used}, "
            f"slots={self._slot_filling_used}, "
            f"intents={self._intent_chain})"
        )
