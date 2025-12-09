"""
LangChain LLM Client - Интеграция с LLM для AI-ассистента аптеки.

================================================================================
РОЛЬ LANGCHAIN В ПРОЕКТЕ
================================================================================

LangChain выступает как "умный мозг" ассистента со следующими задачами:

1. КЛАССИФИКАЦИЯ ИНТЕНТОВ (Intent Classification)
   - Получает сообщение пользователя и контекст
   - Выбирает ОДИН интент из фиксированного списка IntentType
   - Возвращает confidence (0-1) для ансамблевой логики с Router'ом
   
2. ИЗВЛЕЧЕНИЕ СЛОТОВ (Slot Extraction)
   - Структурированное извлечение параметров через Pydantic-модели
   - Поддерживаемые слоты: age, symptom, disease, price_min/max, dosage_form, etc.
   - Использует with_structured_output для гарантии JSON-валидности
   
3. ГЕНЕРАЦИЯ ОТВЕТА (Reply Generation)
   - Формирует человеко-понятный текст на русском языке
   - Соблюдает медицинскую безопасность (не ставит диагнозов)
   - Кратко, вежливо, информативно

4. ДИЗАМБИГУАЦИЯ (Disambiguation)
   - Когда Router не уверен, LLM выбирает из кандидатов
   - Возвращает reasoning для отладки

================================================================================
PIPELINE ВЫЗОВА
================================================================================

Вызовы LangChain происходят в следующих сценариях:

1. router_only - Router уверен (confidence >= 0.85)
   → LangChain НЕ вызывается, используем результат Router'а
   
2. router+slots - Router уверен, но нужно извлечь слоты
   → LangChain вызывается только для slot extraction (дешёвая операция)
   
3. router+llm - Router не уверен, есть кандидаты
   → LangChain дизамбигуирует между кандидатами
   → Возвращает LLMDisambiguationResult
   
4. llm_only - Router ничего не нашёл
   → LangChain полностью определяет intent + slots
   → Возвращает LLMIntentResult

================================================================================
КОНФИГУРАЦИЯ
================================================================================

Температура:
- Intent classification: 0.1-0.2 (минимум галлюцинаций)
- Reply beautification: 0.4-0.5 (немного креатива)

Кэширование:
- LangChain InMemoryCache для повторяющихся запросов
- Кастомный CachingService для более гибкого TTL

LangSmith:
- Трейсинг всех вызовов для отладки
- Метаданные: conversation_id, user_id, intent, pipeline_path
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langsmith import Client as LangSmithClient, traceable
from langchain_core.caches import InMemoryCache
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.exceptions import OutputParserException
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# RunnableRetry location changed across langchain-core versions.
try:  # pragma: no cover - compatibility shim
    from langchain_core.runnables import RunnableRetry
except Exception:  # noqa: BLE001
    class RunnableRetry:  # type: ignore[override]
        """Minimal async retry wrapper for older langchain-core versions."""

        def __init__(self, runnable, max_attempts: int = 2, retry_if_exception_type=()):
            self._runnable = runnable
            self._max_attempts = max_attempts
            self._retry_if_exception_type = retry_if_exception_type or ()

        async def ainvoke(self, *args, **kwargs):
            attempt = 0
            last_exc: Exception | None = None
            while attempt < self._max_attempts:
                try:
                    return await self._runnable.ainvoke(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    attempt += 1
                    if not isinstance(exc, self._retry_if_exception_type):
                        raise
                    if attempt >= self._max_attempts:
                        raise


# Older langchain-core packaged OutputParserException under output_parsers.
try:  # pragma: no cover - compatibility shim
    from langchain_core.output_parsers import OutputParserException as LegacyOutputParserException
except Exception:  # noqa: BLE001
    LegacyOutputParserException = OutputParserException  # type: ignore[misc]
else:
    OutputParserException = LegacyOutputParserException

# LangSmith callbacks are optional; some package versions do not ship them.
try:  # pragma: no cover - import guard for optional dependency
    from langsmith.callbacks import LangChainCallbackHandler
except Exception:  # noqa: BLE001
    LangChainCallbackHandler = None  # type: ignore[assignment]

from ..config import Settings
from ..intents import IntentType
from ..models.assistant import AssistantMeta, AssistantResponse, AssistantAction, Reply
from ..models.llm_intent import (
    ExtractedSlots,
    LLMDisambiguationResult,
    LLMIntentResult,
    LLMSlotExtractionResult,
)
from ..prompts.beautify_prompt import build_beautify_prompt
from ..prompts.intent_prompt import (
    build_intent_prompt,
    build_disambiguation_prompt,
    build_slot_extraction_prompt,
)
from .cache import CachingService, get_caching_service
from .conversation_store import ConversationStore, get_conversation_store
from .dialog_state_store import DialogState

logger = logging.getLogger(__name__)


# =============================================================================
# Константы
# =============================================================================

# Порог уверенности для использования результата Router'а без LLM
ROUTER_CONFIDENCE_THRESHOLD = 0.85

# Температура для разных задач
INTENT_TEMPERATURE = 0.15  # Низкая для предсказуемости
BEAUTIFY_TEMPERATURE = 0.45  # Выше для естественности

# Timeout для LLM вызовов
LLM_TIMEOUT_SECONDS = 30


# =============================================================================
# Data classes для результатов
# =============================================================================

@dataclass
class LLMRunResult:
    """Результат вызова LLM для классификации интента."""
    
    response: AssistantResponse
    token_usage: Dict[str, int] = field(default_factory=dict)
    cached: bool = False
    
    # Расширенные метаданные
    llm_intent_result: Optional[LLMIntentResult] = None
    pipeline_path: str = "llm_only"
    router_confidence: Optional[float] = None
    llm_confidence: Optional[float] = None
    extracted_entities_before: Dict[str, Any] = field(default_factory=dict)
    extracted_entities_after: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BeautifyResult:
    """Результат beautify_reply."""
    
    reply: Reply
    cached: bool = False


@dataclass  
class IntentClassificationResult:
    """Внутренний результат классификации интента."""
    
    intent: IntentType
    confidence: float
    slots: Dict[str, Any]
    reply: str
    reasoning: Optional[str] = None
    needs_clarification: bool = False
    missing_slots: List[str] = field(default_factory=list)


# =============================================================================
# Основной клиент
# =============================================================================

class LangchainLLMClient:
    """
    Клиент LangChain для интеграции с LLM.
    
    Использование:
        client = LangchainLLMClient(settings)
        
        # Полная классификация (intent + slots + reply)
        result = await client.classify_intent(
            message="У меня болит голова",
            router_candidates=None,  # или список кандидатов от Router
            ...
        )
        
        # Дизамбигуация между кандидатами
        result = await client.disambiguate(
            message="...",
            candidates=["FIND_BY_SYMPTOM", "FIND_BY_DISEASE"],
            ...
        )
        
        # Только извлечение слотов
        slots = await client.extract_slots(
            message="...",
            intent=IntentType.FIND_BY_SYMPTOM,
            ...
        )
    """

    def __init__(
        self,
        settings: Settings,
        *,
        llm: ChatOpenAI | None = None,
        cache: CachingService | None = None,
        conversation_store: ConversationStore | None = None,
        history_limit: int = 10,
    ) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for LangChain mode")

        self._settings = settings
        self._enable_llm_cache(settings)
        self._langsmith_client = self._setup_langsmith(settings)
        self._conversation_store = conversation_store or get_conversation_store()
        self._history_limit = history_limit
        callbacks = self._build_callbacks()
        
        # Основная LLM для intent classification (низкая температура)
        if llm is None:
            self._intent_llm_base = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=INTENT_TEMPERATURE,
                timeout=settings.http_timeout_seconds or LLM_TIMEOUT_SECONDS,
                base_url=settings.openai_base_url,
                callbacks=callbacks or None,
            )
        else:
            self._intent_llm_base = llm
            
        # LLM для beautify (повышенная температура)
        self._beautify_llm_base = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=BEAUTIFY_TEMPERATURE,
            timeout=settings.http_timeout_seconds or LLM_TIMEOUT_SECONDS,
            base_url=settings.openai_base_url,
            callbacks=callbacks or None,
        )
        
        self._llm_callbacks = callbacks or []
        self._cache = cache or get_caching_service()
        
        # Промпты
        schema_hint = json.dumps(LLMIntentResult.model_json_schema(), ensure_ascii=False, indent=2)
        self._intent_prompt = build_intent_prompt(schema_hint)
        self._beautify_prompt = build_beautify_prompt()
        
        # Structured output для intent classification
        self._intent_llm = self._intent_llm_base.with_structured_output(
            LLMIntentResult,
            include_raw=True,
            strict=True,
        )
        
        # Structured output для beautify
        self._beautify_llm = self._beautify_llm_base.with_structured_output(
            Reply,
            include_raw=True,
            strict=True,
        )
        
        # Structured output для slot extraction
        self._slot_extraction_llm = self._intent_llm_base.with_structured_output(
            LLMSlotExtractionResult,
            include_raw=True,
            strict=True,
        )
        
        # Structured output для disambiguation
        self._disambiguation_llm = self._intent_llm_base.with_structured_output(
            LLMDisambiguationResult,
            include_raw=True,
            strict=True,
        )
        
        # Retry chains
        self._intent_chain = RunnableRetry(
            self._intent_prompt | self._intent_llm,
            max_attempts=2,
            retry_if_exception_type=(OutputParserException, ValueError),
        )
        
        self._beautify_chain = RunnableRetry(
            self._beautify_prompt | self._beautify_llm,
            max_attempts=2,
            retry_if_exception_type=(OutputParserException, ValueError),
        )
        
        # History-aware chain для intent (если поддерживается)
        if hasattr(self._intent_chain, "with_listeners"):
            self._intent_with_history = RunnableWithMessageHistory(
                self._intent_chain,
                self._history_factory,
                input_messages_key="context_messages",
                history_messages_key="context_messages",
            )
        else:
            self._intent_with_history = self._intent_chain

    # =========================================================================
    # Основные публичные методы
    # =========================================================================

    @traceable(run_type="chain", name="classify_intent")
    async def classify_intent(
        self,
        *,
        message: str,
        profile: Dict[str, Any] | None,
        preference_summary: str | None = None,
        dialog_state: DialogState | None,
        ui_state: Dict[str, Any] | None,
        available_intents: List[str],
        router_candidates: List[Tuple[str, float]] | None = None,
        router_slots: Dict[str, Any] | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> LLMRunResult:
        """
        Классифицирует интент с помощью LLM.
        
        Если router_candidates предоставлены, LLM выбирает из них.
        Иначе LLM самостоятельно определяет интент.
        
        Args:
            message: Сообщение пользователя
            profile: Профиль пользователя
            preference_summary: Резюме предпочтений
            dialog_state: Состояние диалога
            ui_state: Состояние UI
            available_intents: Список доступных интентов
            router_candidates: Кандидаты от Router'а [(intent, confidence), ...]
            router_slots: Слоты, извлечённые Router'ом
            conversation_id: ID разговора
            user_id: ID пользователя
            trace_id: ID трассировки
            
        Returns:
            LLMRunResult с классифицированным интентом
        """
        normalized_message = message.strip().lower()
        profile_signature = json.dumps(profile or {}, ensure_ascii=False, sort_keys=True)
        
        # Проверяем кэш
        cached = self._cache.get_llm_response(normalized_message, profile_signature)
        if cached:
            logger.info(
                "trace_id=%s user_id=%s conversation_id=%s classify_intent cached=True",
                trace_id or "-",
                user_id or "-",
                conversation_id or "-",
            )
            try:
                llm_result = LLMIntentResult.model_validate(cached.get("llm_result", cached))
                response = self._build_assistant_response(llm_result)
                return LLMRunResult(
                    response=response,
                    token_usage={},
                    cached=True,
                    llm_intent_result=llm_result,
                    pipeline_path=cached.get("pipeline_path", "llm_only"),
                    llm_confidence=llm_result.confidence,
                )
            except Exception:
                pass  # Cache miss on validation failure
        
        # Определяем pipeline path
        pipeline_path = "llm_only"
        if router_candidates:
            pipeline_path = "router+llm"
        
        # Формируем payload для LLM
        payload = {
            "message": message,
            "profile_json": profile_signature,
            "preference_summary": preference_summary or "",
            "dialog_state_json": json.dumps(self._dialog_state_to_dict(dialog_state), ensure_ascii=False),
            "ui_state_json": json.dumps(ui_state or {}, ensure_ascii=False),
            "available_intents": ", ".join(sorted(available_intents)),
            "context_messages": [],
        }
        
        # Добавляем кандидатов если есть
        if router_candidates:
            candidates_str = ", ".join([f"{intent}({conf:.2f})" for intent, conf in router_candidates])
            payload["router_candidates"] = candidates_str
        
        # Добавляем предварительно извлечённые слоты
        if router_slots:
            payload["extracted_slots"] = json.dumps(router_slots, ensure_ascii=False)
        
        metadata = self._build_intent_metadata(
            conversation_id=conversation_id,
            user_id=user_id,
            dialog_state=dialog_state,
            available_intents=available_intents,
            ui_state=ui_state,
            pipeline_path=pipeline_path,
        )
        
        config = self._build_invoke_config(conversation_id, metadata)
        
        try:
            if conversation_id:
                result = await self._intent_with_history.ainvoke(payload, config=config)
            else:
                result = await self._intent_chain.ainvoke(payload, config=config)
            
            llm_result, ai_message = self._parse_llm_intent_result(result)
            
            # Мержим слоты от Router'а и LLM
            if router_slots:
                merged_slots = {**router_slots, **llm_result.slots.to_dict()}
                llm_result.slots = ExtractedSlots(**merged_slots)
            
            # Кэшируем результат
            cache_data = {
                "llm_result": llm_result.model_dump(),
                "pipeline_path": pipeline_path,
            }
            self._cache.set_llm_response(
                normalized_message,
                profile_signature,
                cache_data,
                ttl_seconds=900,
            )
            
            response = self._build_assistant_response(llm_result)
            usage = self._extract_usage(ai_message)
            
            logger.info(
                "trace_id=%s user_id=%s conversation_id=%s classify_intent cached=False "
                "intent=%s confidence=%.2f pipeline=%s tokens=%s",
                trace_id or "-",
                user_id or "-",
                conversation_id or "-",
                llm_result.intent.value,
                llm_result.confidence,
                pipeline_path,
                usage,
            )
            
            return LLMRunResult(
                response=response,
                token_usage=usage,
                cached=False,
                llm_intent_result=llm_result,
                pipeline_path=pipeline_path,
                llm_confidence=llm_result.confidence,
                extracted_entities_before=router_slots or {},
                extracted_entities_after=llm_result.slots.to_dict(),
            )
            
        except Exception as exc:  # pragma: no cover - network level failures
            logger.exception("LangChain classify_intent failed: %s", exc)
            fallback = self._fallback_assistant_response(message=message)
            return LLMRunResult(
                response=fallback,
                token_usage={},
                cached=False,
                pipeline_path="llm_only",
            )

    @traceable(run_type="chain", name="disambiguate")
    async def disambiguate(
        self,
        *,
        message: str,
        candidates: List[str],
        dialog_state: DialogState | None = None,
        conversation_id: str | None = None,
        trace_id: str | None = None,
    ) -> LLMDisambiguationResult:
        """
        Дизамбигуирует между кандидатами интентов.
        
        Используется когда Router не уверен и предлагает несколько вариантов.
        
        Args:
            message: Сообщение пользователя
            candidates: Список интентов-кандидатов
            dialog_state: Состояние диалога
            conversation_id: ID разговора
            trace_id: ID трассировки
            
        Returns:
            LLMDisambiguationResult с выбранным интентом
        """
        prompt = build_disambiguation_prompt(candidates)
        llm = self._intent_llm_base.with_structured_output(
            LLMDisambiguationResult,
            include_raw=True,
            strict=True,
        )
        chain = RunnableRetry(
            prompt | llm,
            max_attempts=2,
            retry_if_exception_type=(OutputParserException, ValueError),
        )
        
        payload = {
            "message": message,
            "dialog_state_json": json.dumps(self._dialog_state_to_dict(dialog_state), ensure_ascii=False),
        }
        
        try:
            result = await chain.ainvoke(payload)
            parsed, _ = self._parse_disambiguation_result(result)
            
            logger.info(
                "trace_id=%s disambiguate selected=%s confidence=%.2f reason=%s",
                trace_id or "-",
                parsed.selected_intent.value,
                parsed.confidence,
                parsed.reasoning[:50] if parsed.reasoning else "-",
            )
            
            return parsed
            
        except Exception as exc:
            logger.exception("LangChain disambiguate failed: %s", exc)
            # Fallback: выбираем первого кандидата
            return LLMDisambiguationResult(
                selected_intent=IntentType(candidates[0]) if candidates else IntentType.UNKNOWN,
                confidence=0.5,
                reasoning=f"Fallback due to error: {exc}",
            )

    @traceable(run_type="chain", name="extract_slots")
    async def extract_slots(
        self,
        *,
        message: str,
        intent: IntentType,
        existing_slots: Dict[str, Any] | None = None,
        conversation_id: str | None = None,
        trace_id: str | None = None,
    ) -> ExtractedSlots:
        """
        Извлекает слоты для известного интента.
        
        Используется когда интент уже определён (Router'ом), но нужно
        более качественно извлечь слоты.
        
        Args:
            message: Сообщение пользователя
            intent: Известный интент
            existing_slots: Уже извлечённые слоты
            conversation_id: ID разговора
            trace_id: ID трассировки
            
        Returns:
            ExtractedSlots с извлечёнными параметрами
        """
        prompt = build_slot_extraction_prompt(intent)
        llm = self._intent_llm_base.with_structured_output(
            LLMSlotExtractionResult,
            include_raw=True,
            strict=True,
        )
        chain = RunnableRetry(
            prompt | llm,
            max_attempts=2,
            retry_if_exception_type=(OutputParserException, ValueError),
        )
        
        payload = {
            "message": message,
            "intent": intent.value,
            "existing_slots": json.dumps(existing_slots or {}, ensure_ascii=False),
        }
        
        try:
            result = await chain.ainvoke(payload)
            parsed, _ = self._parse_slot_extraction_result(result)
            
            # Мержим с существующими слотами
            if existing_slots:
                merged = {**existing_slots, **parsed.slots.to_dict()}
                return ExtractedSlots(**merged)
            
            logger.info(
                "trace_id=%s extract_slots intent=%s slots=%s",
                trace_id or "-",
                intent.value,
                list(parsed.slots.to_dict().keys()),
            )
            
            return parsed.slots
            
        except Exception as exc:
            logger.exception("LangChain extract_slots failed: %s", exc)
            return ExtractedSlots(**(existing_slots or {}))

    @traceable(run_type="chain", name="beautify_reply")
    async def beautify_reply(
        self,
        *,
        base_reply: Reply,
        data: Dict[str, Any],
        constraints: Dict[str, Any] | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
        intent: str | None = None,
        router_matched: bool | None = None,
        slot_filling_used: bool | None = None,
        channel: str | None = None,
        metadata: Dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> BeautifyResult:
        """
        Улучшает текст ответа с помощью LLM.
        
        Делает ответ более естественным и дружелюбным.
        """
        # Compute cache keys
        data_hash = self._cache.compute_data_hash(data)
        constraints_hash = self._cache.compute_constraints_hash(constraints)
        
        # Check cache
        cached = self._cache.get_beautify_response(
            base_reply.text,
            data_hash,
            constraints_hash,
        )
        if cached:
            try:
                logger.info(
                    "trace_id=%s user_id=%s conversation_id=%s beautify_reply cached=True",
                    trace_id or "-",
                    user_id or "-",
                    conversation_id or "-",
                )
                return BeautifyResult(reply=Reply.model_validate(cached), cached=True)
            except Exception:
                pass  # Cache miss on validation failure
        
        payload = {
            "base_reply": json.dumps(base_reply.model_dump(), ensure_ascii=False),
            "data_json": json.dumps(data, ensure_ascii=False),
            "constraints_json": json.dumps(constraints or {}, ensure_ascii=False),
        }
        
        combined_metadata = self._build_beautify_metadata(
            conversation_id=conversation_id,
            user_id=user_id,
            intent=intent,
            router_matched=router_matched,
            slot_filling_used=slot_filling_used,
            channel=channel,
            extra=metadata,
        )
        
        try:
            config = self._build_invoke_config(conversation_id, combined_metadata)
            result = await self._beautify_chain.ainvoke(payload, config=config)
            reply, _ = self._parse_reply_json(result)
            
            # Cache the result
            self._cache.set_beautify_response(
                base_reply.text,
                data_hash,
                constraints_hash,
                reply.model_dump(),
                ttl_seconds=600,
            )
            
            logger.info(
                "trace_id=%s user_id=%s conversation_id=%s beautify_reply cached=False",
                trace_id or "-",
                user_id or "-",
                conversation_id or "-",
            )
            
            return BeautifyResult(reply=reply, cached=False)
            
        except Exception as exc:  # pragma: no cover - network level failures
            logger.exception("LangChain beautify_reply failed: %s", exc)
            return BeautifyResult(reply=base_reply, cached=False)

    # =========================================================================
    # Legacy API для обратной совместимости
    # =========================================================================

    async def parse_intent(
        self,
        *,
        message: str,
        profile: Dict[str, Any] | None,
        preference_summary: str | None = None,
        dialog_state: DialogState | None,
        ui_state: Dict[str, Any] | None,
        available_intents: List[str],
        conversation_id: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> LLMRunResult:
        """
        Legacy метод для обратной совместимости.
        
        Переадресует вызов на classify_intent.
        """
        return await self.classify_intent(
            message=message,
            profile=profile,
            preference_summary=preference_summary,
            dialog_state=dialog_state,
            ui_state=ui_state,
            available_intents=available_intents,
            conversation_id=conversation_id,
            user_id=user_id,
            trace_id=trace_id,
        )

    # =========================================================================
    # Приватные методы - парсинг результатов
    # =========================================================================

    def _parse_llm_intent_result(self, result: Any) -> Tuple[LLMIntentResult, AIMessage | None]:
        """Парсит результат LLM в LLMIntentResult."""
        parsed = result
        raw_message: AIMessage | None = None
        
        if isinstance(result, dict) and "parsed" in result:
            parsed = result.get("parsed")
            raw_message = result.get("raw")
        elif hasattr(result, "raw"):
            raw_message = getattr(result, "raw", None)
            parsed = getattr(result, "parsed", result)
        
        if isinstance(parsed, LLMIntentResult):
            return parsed, raw_message
        
        if isinstance(parsed, AIMessage):
            raw_message = parsed
            content = _extract_message_content(parsed)
            try:
                return LLMIntentResult.model_validate_json(content), raw_message
            except Exception as exc:
                logger.warning("Failed to parse LLMIntentResult JSON: %s; payload=%s", exc, content[:200])
                return self._fallback_llm_intent_result(), raw_message
        
        try:
            return LLMIntentResult.model_validate(parsed), raw_message
        except Exception as exc:
            logger.warning("Failed to parse LLMIntentResult: %s; payload=%s", exc, parsed)
            return self._fallback_llm_intent_result(), raw_message

    def _parse_disambiguation_result(self, result: Any) -> Tuple[LLMDisambiguationResult, AIMessage | None]:
        """Парсит результат дизамбигуации."""
        parsed = result
        raw_message: AIMessage | None = None
        
        if isinstance(result, dict) and "parsed" in result:
            parsed = result.get("parsed")
            raw_message = result.get("raw")
        elif hasattr(result, "raw"):
            raw_message = getattr(result, "raw", None)
            parsed = getattr(result, "parsed", result)
        
        if isinstance(parsed, LLMDisambiguationResult):
            return parsed, raw_message
        
        try:
            return LLMDisambiguationResult.model_validate(parsed), raw_message
        except Exception as exc:
            logger.warning("Failed to parse LLMDisambiguationResult: %s", exc)
            return LLMDisambiguationResult(
                selected_intent=IntentType.UNKNOWN,
                confidence=0.5,
                reasoning="Parse error",
            ), raw_message

    def _parse_slot_extraction_result(self, result: Any) -> Tuple[LLMSlotExtractionResult, AIMessage | None]:
        """Парсит результат извлечения слотов."""
        parsed = result
        raw_message: AIMessage | None = None
        
        if isinstance(result, dict) and "parsed" in result:
            parsed = result.get("parsed")
            raw_message = result.get("raw")
        elif hasattr(result, "raw"):
            raw_message = getattr(result, "raw", None)
            parsed = getattr(result, "parsed", result)
        
        if isinstance(parsed, LLMSlotExtractionResult):
            return parsed, raw_message
        
        try:
            return LLMSlotExtractionResult.model_validate(parsed), raw_message
        except Exception as exc:
            logger.warning("Failed to parse LLMSlotExtractionResult: %s", exc)
            return LLMSlotExtractionResult(
                slots=ExtractedSlots(),
                confidence=0.5,
            ), raw_message

    def _parse_reply_json(self, result: Any) -> Tuple[Reply, AIMessage | None]:
        """Парсит результат beautify."""
        parsed = result
        raw_message: AIMessage | None = None
        
        if isinstance(result, dict) and "parsed" in result:
            parsed = result.get("parsed")
            raw_message = result.get("raw")
        elif hasattr(result, "raw"):
            raw_message = getattr(result, "raw", None)
            parsed = getattr(result, "parsed", result)
        
        if isinstance(parsed, Reply):
            return parsed, raw_message
        
        if isinstance(parsed, AIMessage):
            raw_message = parsed
            content = _extract_message_content(parsed)
            try:
                return Reply.model_validate_json(content), raw_message
            except Exception as exc:
                logger.warning("Failed to parse Reply JSON: %s; payload=%s", exc, content[:200])
                return Reply(text=content if isinstance(content, str) else "Готово."), raw_message
        
        try:
            return Reply.model_validate(parsed), raw_message
        except Exception as exc:
            logger.warning("Failed to parse Reply: %s; payload=%s", exc, parsed)
            return Reply(text=str(parsed) if parsed else "Готово."), raw_message

    # =========================================================================
    # Приватные методы - построение ответов
    # =========================================================================

    def _build_assistant_response(self, llm_result: LLMIntentResult) -> AssistantResponse:
        """Строит AssistantResponse из LLMIntentResult."""
        from ..intents import ActionType, ActionChannel
        
        # Определяем channel по интенту
        channel = self._get_channel_for_intent(llm_result.intent)
        
        # Формируем action
        action = AssistantAction(
            type=ActionType.CALL_PLATFORM_API,
            intent=llm_result.intent,
            channel=channel,
            parameters=llm_result.slots.to_dict(),
        )
        
        meta = AssistantMeta(
            top_intent=llm_result.intent.value,
            confidence=llm_result.confidence,
            extracted_entities=llm_result.slots.to_dict(),
            debug={
                "llm_used": True,
                "llm_confidence": llm_result.confidence,
                "reasoning": llm_result.reasoning,
                "needs_clarification": llm_result.needs_clarification,
                "missing_slots": llm_result.missing_required_slots,
            },
        )
        
        return AssistantResponse(
            reply=Reply(text=llm_result.reply),
            actions=[action],
            meta=meta,
        )

    def _get_channel_for_intent(self, intent: IntentType) -> str:
        """Определяет channel для интента."""
        from ..intents import NAVIGATION_INTENTS, ORDER_INTENTS, SYMPTOM_INTENTS
        
        if intent in NAVIGATION_INTENTS:
            return "navigation"
        if intent in ORDER_INTENTS:
            return "order"
        if intent in SYMPTOM_INTENTS:
            return "data"
        return "data"

    @staticmethod
    def _fallback_llm_intent_result() -> LLMIntentResult:
        """Возвращает fallback результат при ошибке парсинга."""
        return LLMIntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.3,
            slots=ExtractedSlots(),
            reply="Не удалось обработать запрос. Пожалуйста, переформулируйте вопрос.",
            reasoning="Fallback due to parse error",
        )

    @staticmethod
    def _fallback_assistant_response(message: str | None = None) -> AssistantResponse:
        """Возвращает fallback AssistantResponse при ошибке."""
        reply_text = (
            "Не удалось обработать запрос автоматически. "
            "Пожалуйста, переформулируйте вопрос или выберите подсказку ниже."
        )
        if message:
            reply_text += f" Последнее сообщение: \"{message.strip()[:50]}\"."
        
        return AssistantResponse(
            reply=Reply(text=reply_text),
            actions=[],
            meta=AssistantMeta(
                confidence=0.3,
                debug={"llm_used": True, "fallback": True},
            ),
        )

    # =========================================================================
    # Приватные методы - конфигурация и утилиты
    # =========================================================================

    @staticmethod
    def _setup_langsmith(settings: Settings) -> LangSmithClient | None:
        """Initialize LangSmith tracing when enabled via settings/env."""
        tracing_enabled = bool(settings.langsmith_tracing_v2 or settings.langchain_tracing)
        if tracing_enabled:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        if not (settings.langsmith_api_key and settings.langsmith_tracing_v2):
            return None
        try:
            project = settings.langsmith_project or "smart-pharmacy-assistant"
            os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
            os.environ.setdefault("LANGSMITH_PROJECT", project)
            if settings.langsmith_endpoint:
                os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
                os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.langsmith_endpoint)
            else:
                os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            return LangSmithClient(
                api_key=settings.langsmith_api_key,
                api_url=settings.langsmith_endpoint or None,
                project=project,
            )
        except Exception as exc:  # pragma: no cover - tracing is best-effort
            logger.warning("LangSmith client disabled: %s", exc)
            return None

    @staticmethod
    def _enable_llm_cache(settings: Settings) -> None:
        """Enable LangChain in-memory cache if configured."""
        if not settings.langchain_cache:
            return
        set_llm_cache(InMemoryCache())

    @staticmethod
    def _build_invoke_config(
        conversation_id: str | None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        config: Dict[str, Any] = {}
        if conversation_id:
            config["configurable"] = {"session_id": conversation_id}
        if metadata:
            config["metadata"] = metadata
        return config or None

    def _build_callbacks(self) -> list:
        """Return LangSmith callbacks when tracing is configured."""
        if not self._langsmith_client:
            return []
        if LangChainCallbackHandler is None:
            logger.warning("LangSmith callback handler unavailable; tracing callbacks disabled")
            return []
        try:
            return [LangChainCallbackHandler(client=self._langsmith_client)]
        except Exception as exc:  # pragma: no cover - optional tracing
            logger.warning("Failed to initialize LangSmith callback handler: %s", exc)
            return []

    def _history_factory(self, session_id: str) -> BaseChatMessageHistory:
        return _ConversationStoreHistory(self._conversation_store, session_id, self._history_limit)

    @staticmethod
    def _extract_usage(message: AIMessage | None) -> Dict[str, int]:
        if message is None:
            return {}
        metadata = getattr(message, "response_metadata", {}) or {}
        usage = metadata.get("token_usage") or {}
        return {k: int(v) for k, v in usage.items() if isinstance(v, (int, float))}

    @staticmethod
    def _dialog_state_to_dict(dialog_state: DialogState | None) -> Dict[str, Any]:
        if dialog_state is None:
            return {}
        return {
            "intent": getattr(dialog_state.current_intent, "value", dialog_state.current_intent),
            "channel": getattr(dialog_state.channel, "value", dialog_state.channel),
            "slots": dialog_state.slots,
            "context_products": dialog_state.context_products,
            "last_reply": dialog_state.last_reply,
        }

    @staticmethod
    def _build_intent_metadata(
        *,
        conversation_id: str | None,
        user_id: str | None,
        dialog_state: DialogState | None,
        available_intents: List[str],
        ui_state: Dict[str, Any] | None,
        pipeline_path: str = "llm_only",
    ) -> Dict[str, Any]:
        metadata = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "channel": getattr(dialog_state, "channel", None),
            "intent": getattr(dialog_state, "current_intent", None),
            "pipeline_path": pipeline_path,
            "available_intents": sorted(available_intents),
        }
        if ui_state:
            metadata["ui_state"] = {k: v for k, v in ui_state.items() if v is not None}
        return LangchainLLMClient._normalize_metadata(metadata)

    @staticmethod
    def _build_beautify_metadata(
        *,
        conversation_id: str | None,
        user_id: str | None,
        intent: str | None,
        router_matched: bool | None,
        slot_filling_used: bool | None,
        channel: str | None,
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        metadata = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "intent": intent,
            "router_matched": router_matched,
            "slot_filling_used": slot_filling_used,
            "channel": channel,
        }
        if extra:
            metadata.update(extra)
        return LangchainLLMClient._normalize_metadata(metadata)

    @staticmethod
    def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if hasattr(value, "value"):
                normalized[key] = getattr(value, "value")
            else:
                normalized[key] = value
        return normalized


# =============================================================================
# Helper classes
# =============================================================================

def _extract_message_content(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        # OpenAI can return list[dict]; join textual segments
        return " ".join(
            part.get("text", "")
            for part in message.content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(message.content)


class _ConversationStoreHistory(BaseChatMessageHistory):
    """Adapter to expose ConversationStore as LangChain message history."""

    def __init__(self, store: ConversationStore, session_id: str, limit: int = 10) -> None:
        self._store = store
        self._session_id = session_id or ""
        self._limit = limit

    @property
    def messages(self) -> list[BaseMessage]:
        history = self._store.get_history(self._session_id, limit=self._limit)
        result: list[BaseMessage] = []
        for item in history:
            if item.role == "assistant":
                result.append(AIMessage(content=item.content))
            else:
                result.append(HumanMessage(content=item.content))
        return result

    def add_message(self, message: BaseMessage) -> None:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        self._store.append_message(self._session_id, role, str(message.content))

    def clear(self) -> None:
        # No-op; clearing is managed outside this adapter.
        return
