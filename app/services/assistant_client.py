"""
AssistantClient - Высокоуровневый клиент для работы с LLM.

================================================================================
РОЛЬ В АРХИТЕКТУРЕ
================================================================================

AssistantClient служит связующим звеном между Router'ом и LangChain LLM:

1. Получает запрос от основного пайплайна
2. Определяет нужно ли вызывать LLM (на основе confidence Router'а)
3. Вызывает LangchainLLMClient с правильными параметрами
4. Обновляет профиль пользователя на основе извлечённых предпочтений
5. Управляет историей разговора

================================================================================
ИНТЕГРАЦИЯ С АНСАМБЛЕМ ROUTER + LLM
================================================================================

Поддерживает три режима работы:

1. router_only / router+slots:
   - Router уверен, LLM не вызывается для классификации
   - LLM может вызываться только для beautify_reply

2. router+llm:
   - Router не уверен, есть кандидаты
   - LLM дизамбигуирует между кандидатами
   - Вызывается classify_intent с router_candidates

3. llm_only:
   - Router ничего не нашёл
   - LLM полностью определяет intent + slots
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from ..config import Settings
from ..intents import IntentType
from ..models import ChatRequest, DataPayload, UserProfile
from ..models.assistant import AssistantMeta, AssistantResponse, Reply
from ..models.llm_intent import ExtractedSlots, LLMIntentResult
from .conversation_store import ConversationStore, get_conversation_store
from .dialog_state_store import DialogStateStore, get_dialog_state_store
from .debug_meta import DebugMetaBuilder
from .errors import LLMError
from .langchain_llm import LangchainLLMClient, LLMRunResult
from .metrics import get_metrics_service
from .user_profile_store import UserProfileStore, get_user_profile_store
from ..utils.logging import get_request_logger

logger = logging.getLogger(__name__)


class AssistantClientError(LLMError):
    """Raised when the assistant integration fails."""


_PREFERENCE_KEYWORDS: Dict[str, List[str]] = {
    "sugar_free": ["без сахара", "сахар не добавляй", "sugar-free", "sugar free"],
    "lactose_free": ["без лактозы", "lactose-free", "lactose free"],
    "for_children": [
        "для ребенка",
        "для ребёнка",
        "для детей",
        "детский",
        "kid",
        "у меня дети",
        "есть дети",
        "детям",
        "сын",
        "дочь",
    ],
}

_FORM_KEYWORDS: Dict[str, List[str]] = {
    "спрей": ["спрей", "spray"],
    "таблетки": ["таблет", "pill"],
    "сироп": ["сироп", "syrup"],
    "капли": ["капли", "drops"],
}

_AGE_PATTERN = re.compile(r"(?:мне\s+)?(?P<age>\d{1,2})\s*(?:лет|года|год)", re.IGNORECASE)
_PRICE_PATTERN = re.compile(r"(?:до|не дороже)\s*(?P<price>\d{2,5})\s*(?:р|руб|₽)", re.IGNORECASE)


class AssistantClient:
    """High-level client orchestrating LangChain-based LLM calls."""

    def __init__(
        self,
        settings: Settings,
        conversation_store: ConversationStore | None = None,
        user_profile_store: UserProfileStore | None = None,
        dialog_state_store: DialogStateStore | None = None,
        langchain_client: LangchainLLMClient | None = None,
    ) -> None:
        self._settings = settings
        self._conversation_store = conversation_store or get_conversation_store()
        self._user_profile_store = user_profile_store or get_user_profile_store()
        self._dialog_state_store = dialog_state_store or get_dialog_state_store()
        self._langchain_client: LangchainLLMClient | None = None
        if langchain_client is not None:
            self._langchain_client = langchain_client
        elif self._settings.use_langchain:
            try:
                self._langchain_client = LangchainLLMClient(self._settings)
                logger.info("LangChain LLM client enabled.")
            except ValueError as exc:
                logger.warning("LangChain client unavailable: %s. Falling back to deterministic responses.", exc)
        self._metrics = get_metrics_service()
        self._history_pairs = 4
        self._history_char_cap = 1200
        self._summary_min_messages = 10
        self._summary_refresh_interval = 6
        self._summary_history_limit = 20

    async def analyze_message(
        self,
        request: ChatRequest,
        intents: List[str],
        *,
        router_candidates: List[Tuple[str, float]] | None = None,
        router_slots: Dict[str, Any] | None = None,
        debug_builder: DebugMetaBuilder | None = None,
        trace_id: str | None = None,
    ) -> AssistantResponse:
        """
        Анализирует сообщение с помощью LLM.
        
        Поддерживает режим ансамбля Router + LLM:
        - Если router_candidates предоставлены, LLM выбирает из них
        - Иначе LLM самостоятельно определяет интент
        
        Args:
            request: Запрос пользователя
            intents: Список доступных интентов
            router_candidates: Кандидаты от Router'а [(intent, confidence), ...]
            router_slots: Слоты, извлечённые Router'ом
            debug_builder: Билдер для debug метаданных
            trace_id: ID трассировки
            
        Returns:
            AssistantResponse с классифицированным интентом
        """
        conversation_id = request.conversation_id or ""
        request_logger = get_request_logger(
            logger,
            trace_id=trace_id,
            user_id=request.user_id,
            conversation_id=conversation_id,
        )
        request_logger.info("User message received: %s", request.message[:100])
        
        if not self._settings.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY is not set; returning fallback for conversation_id=%s",
                conversation_id,
            )
            fallback = self._build_fallback_response(request, reason="missing_api_key")
            self._record_exchange(conversation_id, request.message, fallback.reply.text)
            await self._maybe_refresh_summary(conversation_id)
            if debug_builder:
                debug_builder.set_llm_used(False).set_llm_cached(False)
                debug_builder.set_pipeline_path("fallback")
            request_logger.warning("LLM unavailable (no API key), using fallback backend")
            return fallback

        user_profile: UserProfile | None = None
        if request.user_id:
            self._auto_capture_preferences(request.user_id, request.message)
            user_profile = self._user_profile_store.get_or_create(request.user_id)
        preference_summary = self._build_preference_summary(user_profile)

        dialog_state = self._dialog_state_store.get_state(conversation_id)

        if self._langchain_client:
            try:
                # Определяем pipeline path
                pipeline_path = "llm_only"
                if router_candidates:
                    pipeline_path = "router+llm"
                
                llm_result = await self._langchain_client.classify_intent(
                    message=request.message,
                    profile=user_profile.model_dump() if user_profile else None,
                    preference_summary=preference_summary,
                    dialog_state=dialog_state,
                    ui_state=request.ui_state.model_dump() if request.ui_state else None,
                    available_intents=intents,
                    router_candidates=router_candidates,
                    router_slots=router_slots,
                    conversation_id=conversation_id,
                    user_id=request.user_id,
                    trace_id=trace_id,
                )
                
                assistant_response = llm_result.response
                self._metrics.record_llm_call(
                    user_id=request.user_id,
                    cached=llm_result.cached,
                    token_usage=llm_result.token_usage,
                )
                
                if debug_builder:
                    debug_builder.set_llm_used(True, cached=llm_result.cached)
                    debug_builder.set_pipeline_path(llm_result.pipeline_path)
                    
                    if llm_result.llm_confidence is not None:
                        debug_builder.set_llm_confidence(llm_result.llm_confidence)
                    
                    if llm_result.llm_intent_result:
                        if llm_result.llm_intent_result.reasoning:
                            debug_builder.set_llm_reasoning(llm_result.llm_intent_result.reasoning)
                    
                    if llm_result.extracted_entities_before:
                        debug_builder.set_extracted_entities_before(llm_result.extracted_entities_before)
                    if llm_result.extracted_entities_after:
                        debug_builder.set_extracted_entities_after(llm_result.extracted_entities_after)
                    
                    if router_candidates:
                        debug_builder.set_router_candidates([
                            {"intent": i, "confidence": c} for i, c in router_candidates
                        ])
                else:
                    self._attach_llm_debug(
                        assistant_response,
                        used=not llm_result.cached,
                        backend="langchain",
                        cached=llm_result.cached,
                        pipeline_path=llm_result.pipeline_path,
                        llm_confidence=llm_result.llm_confidence,
                    )
                
                self._record_exchange(conversation_id, request.message, assistant_response.reply.text)
                await self._maybe_refresh_summary(conversation_id)
                
                intents_found = [
                    getattr(action.intent, "value", action.intent) 
                    for action in assistant_response.actions
                ]
                request_logger.info(
                    "LLM classify_intent completed backend=langchain cached=%s "
                    "pipeline=%s intents=%s",
                    llm_result.cached,
                    llm_result.pipeline_path,
                    intents_found,
                )
                return assistant_response
                
            except Exception as exc:  # pragma: no cover - safety fallback
                logger.warning(
                    "LangChain client failed for conversation_id=%s; error=%s",
                    conversation_id,
                    exc,
                )

        fallback = self._build_fallback_response(request, reason="missing_llm")
        self._record_exchange(conversation_id, request.message, fallback.reply.text)
        await self._maybe_refresh_summary(conversation_id)
        if debug_builder:
            debug_builder.set_llm_used(False).set_llm_cached(False)
            debug_builder.set_pipeline_path("fallback")
        request_logger.warning("LLM classify_intent failed or unavailable, backend=fallback")
        return fallback

    async def beautify_reply(
        self,
        *,
        reply: Reply,
        data: DataPayload,
        constraints: Dict[str, Any] | None = None,
        user_message: str | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
        intent: str | None = None,
        router_matched: bool | None = None,
        slot_filling_used: bool | None = None,
        channel: str | None = None,
        trace_metadata: Dict[str, Any] | None = None,
        debug_builder: DebugMetaBuilder | None = None,
    ) -> Reply:
        if not self._settings.openai_api_key:
            self._metrics.record_beautify_skipped()
            return reply
        if not self._langchain_client:
            self._metrics.record_beautify_skipped()
            return reply
        constraint_payload: Dict[str, Any] = {}
        if user_message:
            constraint_payload["user_message"] = user_message
        if constraints:
            constraint_payload.update(constraints)
        try:
            result = await self._langchain_client.beautify_reply(
                base_reply=reply,
                data=data.model_dump(),
                constraints=constraint_payload,
                conversation_id=conversation_id,
                user_id=user_id,
                intent=intent,
                router_matched=router_matched,
                slot_filling_used=slot_filling_used,
                channel=channel,
                metadata=trace_metadata,
            )
            cached = getattr(result, "cached", False)
            reply_obj = getattr(result, "reply", result)
            self._metrics.record_beautify_call(cached=cached)
            if debug_builder:
                debug_builder.set_llm_used(True, cached=cached)
            return reply_obj
        except Exception as exc:  # pragma: no cover - network/network issues
            logger.warning("Beautify reply failed: %s", exc)
            self._metrics.record_beautify_skipped()
            return reply

    async def explain_data(self, *, original_reply: Reply, data: DataPayload, user_message: str) -> Reply | None:
        """Backward-compatible helper for tests; delegates to beautify_reply when data has content."""

        if not data.has_content():
            return None
        polished = await self.beautify_reply(
            reply=original_reply,
            data=data,
            constraints={"user_message": user_message},
            user_message=user_message,
        )
        if polished == original_reply:
            return None
        return polished

    def _build_fallback_response(self, request: ChatRequest, *, reason: str = "missing_api_key") -> AssistantResponse:
        """Return a deterministic reply when OpenAI access is unavailable."""

        preview = request.message.strip()
        if len(preview) > 80:
            preview = preview[:77] + "..."
        last_message = preview or "пустое сообщение"
        messages = {
            "missing_api_key": (
                "Ассистент работает в демо-режиме (OPENAI_API_KEY не задан). "
                "Чтобы включить живой ответ, установите переменную окружения OPENAI_API_KEY. "
            ),
            "geo_blocked": (
                "Ассистент временно недоступен: сервис OpenAI не поддерживается в текущем регионе. "
                "Мы показываем технический ответ без живой консультации. "
            ),
            "missing_llm": (
                "Ассистент работает в офлайн-режиме. "
                "Мы обработаем запрос через встроенные сервисы или предложим уточнение. "
            ),
        }
        prefix = messages.get(
            reason,
            "Ассистент временно недоступен. Мы уже работаем над восстановлением сервиса. ",
        )
        text = f'{prefix}Последнее сообщение пользователя: "{last_message}".'
        response = AssistantResponse(reply=Reply(text=text), actions=[])
        self._attach_llm_debug(response, used=False, backend=None, cached=False)
        return response

    def _record_exchange(self, conversation_id: str, user_message: str, assistant_text: str | None) -> None:
        if not conversation_id:
            return
        self._conversation_store.append_message(conversation_id, "user", user_message)
        if assistant_text:
            self._conversation_store.append_message(conversation_id, "assistant", assistant_text)

    async def _maybe_refresh_summary(self, conversation_id: str) -> None:
        return  # Summaries disabled when operating without direct LLM access.

    def _auto_capture_preferences(self, user_id: str, message: str) -> None:
        updates = self._extract_preference_updates(message)
        if updates:
            self._user_profile_store.update_preferences(user_id, **updates)

    def _extract_preference_updates(self, message: str) -> Dict[str, Any]:
        text = message.lower()
        updates: Dict[str, Any] = {}
        for field, keywords in _PREFERENCE_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                updates[field] = True
        preferred_forms: List[str] = []
        for form, keywords in _FORM_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                preferred_forms.append(form)
        if preferred_forms:
            updates["preferred_forms"] = preferred_forms
        age_match = _AGE_PATTERN.search(message)
        if age_match:
            try:
                updates["age"] = int(age_match.group("age"))
            except ValueError:
                pass
        price_match = _PRICE_PATTERN.search(message)
        if price_match:
            try:
                updates["default_max_price"] = float(price_match.group("price"))
            except ValueError:
                pass
        return updates

    def _build_preference_summary(self, profile: UserProfile | None) -> str:
        """Human-readable summary for prompt conditioning."""

        if not profile or not profile.preferences:
            return ""
        prefs = profile.preferences
        parts: list[str] = []
        if prefs.age:
            parts.append(f"{prefs.age} лет")
        preferred_forms = prefs.preferred_forms or []
        if preferred_forms:
            forms = ", ".join(str(form).strip().lower() for form in preferred_forms if str(form).strip())
            if forms:
                parts.append(f"предпочитает {forms}")
        if prefs.default_max_price is not None:
            parts.append(f"до {self._format_price(prefs.default_max_price)}")
        if prefs.sugar_free:
            parts.append("без сахара")
        if prefs.lactose_free:
            parts.append("без лактозы")
        if prefs.for_children:
            parts.append("для детей")
        return ", ".join(parts)

    @staticmethod
    def _format_price(price: float) -> str:
        try:
            value = float(price)
        except (TypeError, ValueError):
            return str(price)
        if value.is_integer():
            return f"{int(value)}₽"
        return f"{value:.2f}₽"

    def _attach_llm_debug(
        self,
        response: AssistantResponse,
        *,
        used: bool,
        backend: str | None,
        cached: bool,
        pipeline_path: str | None = None,
        llm_confidence: float | None = None,
    ) -> None:
        """Добавляет debug информацию о LLM в response.meta."""
        meta = response.meta or AssistantMeta()
        debug_payload = dict(meta.debug or {})
        debug_payload.update(
            {
                "llm_used": used,
                "llm_backend": backend if used else None,
                "llm_cached": cached,
            }
        )
        if pipeline_path:
            debug_payload["pipeline_path"] = pipeline_path
        if llm_confidence is not None:
            debug_payload["llm_confidence"] = round(llm_confidence, 2)
        meta.debug = debug_payload
        response.meta = meta

