from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from ..config import Settings
from ..models import ChatRequest, DataPayload, UserProfile
from ..models.assistant import AssistantMeta, AssistantResponse, Reply
from .conversation_store import ConversationStore, get_conversation_store
from .dialog_state_store import DialogStateStore, get_dialog_state_store
from .debug_meta import DebugMetaBuilder
from .error_handling import LLMError
from .langchain_llm import LangchainLLMClient
from .logging_utils import log_info, log_warning
from .metrics import get_metrics_service
from .user_profile_store import UserProfileStore, get_user_profile_store

logger = logging.getLogger(__name__)


class AssistantClientError(LLMError):
    """Raised when the assistant integration fails."""


_PREFERENCE_KEYWORDS: Dict[str, List[str]] = {
    "sugar_free": ["без сахара", "сахар не добавляй", "sugar-free", "sugar free"],
    "lactose_free": ["без лактозы", "lactose-free", "lactose free"],
    "for_children": ["для ребенка", "для ребёнка", "для детей", "детский", "kid"],
    "has_children": ["у меня дети", "есть дети", "двое детей", "сын", "дочь"],
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
        debug_builder: DebugMetaBuilder | None = None,
        trace_id: str | None = None,
    ) -> AssistantResponse:
        """Send the user message to ChatGPT and parse the structured reply."""

        conversation_id = request.conversation_id or ""
        if not self._settings.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY is not set; returning fallback assistant response for conversation_id=%s",
                conversation_id,
            )
            fallback = self._build_fallback_response(request, reason="missing_api_key")
            self._record_exchange(conversation_id, request.message, fallback.reply.text)
            await self._maybe_refresh_summary(conversation_id)
            if debug_builder:
                debug_builder.set_llm_used(False).set_llm_cached(False)
            log_warning(
                logger,
                "LLM unavailable (no API key), returning fallback response",
                trace_id=trace_id,
                user_id=request.user_id,
                conversation_id=conversation_id,
            )
            return fallback

        user_profile: UserProfile | None = None
        if request.user_id:
            self._auto_capture_preferences(request.user_id, request.message)
            user_profile = self._user_profile_store.get_or_create(request.user_id)

        dialog_state = self._dialog_state_store.get_state(conversation_id)

        if self._langchain_client:
            try:
                llm_result = await self._langchain_client.parse_intent(
                    message=request.message,
                    profile=user_profile.model_dump() if user_profile else None,
                    dialog_state=dialog_state,
                    ui_state=request.ui_state.model_dump() if request.ui_state else None,
                    available_intents=intents,
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
                else:
                    self._attach_llm_debug(
                        assistant_response,
                        used=not llm_result.cached,
                        backend="langchain",
                        cached=llm_result.cached,
                    )
                self._record_exchange(conversation_id, request.message, assistant_response.reply.text)
                await self._maybe_refresh_summary(conversation_id)
                log_info(
                    logger,
                    "LLM parse_intent completed",
                    trace_id=trace_id,
                    user_id=request.user_id,
                    conversation_id=conversation_id,
                    intent=assistant_response.actions[0].intent if assistant_response.actions else None,
                )
                return assistant_response
            except Exception as exc:  # pragma: no cover - safety fallback
                logger.warning(
                    "LangChain client failed for conversation_id=%s; falling back to deterministic reply. error=%s",
                    conversation_id,
                    exc,
                )

        fallback = self._build_fallback_response(request, reason="missing_llm")
        self._record_exchange(conversation_id, request.message, fallback.reply.text)
        await self._maybe_refresh_summary(conversation_id)
        if debug_builder:
            debug_builder.set_llm_used(False).set_llm_cached(False)
        log_warning(
            logger,
            "LLM parse_intent failed or unavailable, using fallback",
            trace_id=trace_id,
            user_id=request.user_id,
            conversation_id=conversation_id,
        )
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
                if field == "for_children":
                    updates.setdefault("has_children", True)
        preferred_forms: List[str] = []
        for form, keywords in _FORM_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                preferred_forms.append(form)
        if preferred_forms:
            updates["preferred_dosage_forms"] = preferred_forms
        age_match = _AGE_PATTERN.search(message)
        if age_match:
            try:
                updates["age"] = int(age_match.group("age"))
            except ValueError:
                pass
        price_match = _PRICE_PATTERN.search(message)
        if price_match:
            try:
                updates["default_max_price"] = int(price_match.group("price"))
            except ValueError:
                pass
        return updates

    def _attach_llm_debug(
        self,
        response: AssistantResponse,
        *,
        used: bool,
        backend: str | None,
        cached: bool,
    ) -> None:
        meta = response.meta or AssistantMeta()
        debug_payload = dict(meta.debug or {})
        debug_payload.update(
            {
                "llm_used": used,
                "llm_backend": backend if used else None,
                "llm_cached": cached,
            }
        )
        meta.debug = debug_payload
        response.meta = meta

