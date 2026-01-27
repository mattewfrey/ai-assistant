from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..config import Settings
from ..models.assistant import Reply
from ..models.product_chat import (
    ProductChatCitation,
    ProductChatMeta,
    ProductChatRefusalReason,
    ProductChatRequest,
    ProductChatResponse,
)
from .audit_service import AuditService, get_audit_service
from .cache import CachingService, get_caching_service
from .conversation_store import ConversationStore, get_conversation_store
from .metrics import MetricsService, get_metrics_service
from .product_chat_session_store import ProductChatSessionStore, get_product_chat_session_store
from .product_context_builder import ProductContextBuilder
from .product_policy_guard import ProductPolicyGuard
from .errors import BadRequestError

logger = logging.getLogger(__name__)


class ProductChatService:
    _PATH_INDEX_RE = re.compile(r"\[(\d+)\]")

    def __init__(
        self,
        *,
        settings: Settings,
        context_builder: ProductContextBuilder,
        policy_guard: ProductPolicyGuard,
        llm_client: Any | None,
        session_store: ProductChatSessionStore | None = None,
        conversation_store: ConversationStore | None = None,
        metrics: MetricsService | None = None,
        audit: AuditService | None = None,
        cache: CachingService | None = None,
    ) -> None:
        self._settings = settings
        self._context_builder = context_builder
        self._policy_guard = policy_guard
        self._llm_client = llm_client
        self._session_store = session_store or get_product_chat_session_store()
        self._conversation_store = conversation_store or get_conversation_store()
        self._metrics = metrics or get_metrics_service()
        self._audit = audit or get_audit_service()
        self._cache = cache or get_caching_service()

    async def handle(
        self,
        *,
        request: ProductChatRequest,
        authorization: Optional[str],
        trace_id: Optional[str],
    ) -> ProductChatResponse:
        if not request.message or not request.message.strip():
            raise BadRequestError("message must not be empty", reason="empty_message")
        if not request.product_id or not request.product_id.strip():
            raise BadRequestError("product_id must not be empty", reason="empty_product_id")

        conversation_id = request.conversation_id or str(uuid4())
        existing_product = self._session_store.get_product_id(conversation_id)
        if existing_product and existing_product != request.product_id:
            raise BadRequestError(
                "conversation_id is bound to a different product",
                reason="product_mismatch",
            )
        if not existing_product:
            self._session_store.set_product_id(conversation_id, request.product_id)

        logger.info(
            "product_chat.handle conversation_id=%s product_id=%s user_id=%s",
            conversation_id,
            request.product_id,
            request.user_id or "-",
        )

        store_id = request.ui_state.store_id if request.ui_state else None
        shipping_method = request.ui_state.shipping_method if request.ui_state else None

        context_result = await self._context_builder.get_context(
            product_id=request.product_id,
            store_id=store_id,
            shipping_method=shipping_method,
            authorization=authorization,
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=request.user_id,
        )
        context = context_result.context
        context_hash = context_result.context_hash
        context_cache_hit = context_result.cache_hit

        # Audit: log incoming request
        self._audit.log_chat_request(
            trace_id=trace_id or conversation_id,
            conversation_id=conversation_id,
            user_id=request.user_id,
            product_id=request.product_id,
            message=request.message,
        )

        policy_decision = self._policy_guard.evaluate(
            request.message, context, conversation_id=conversation_id
        )
        if policy_decision.blocked:
            # Audit: log policy block
            self._audit.log_policy_block(
                trace_id=trace_id or conversation_id,
                conversation_id=conversation_id,
                user_id=request.user_id,
                product_id=request.product_id,
                message=request.message,
                refusal_reason=policy_decision.refusal_reason.value if policy_decision.refusal_reason else "BLOCKED",
                out_of_scope=policy_decision.out_of_scope,
                injection_detected=policy_decision.injection_detected,
            )
            
            reply_text = policy_decision.reply_text or "Я могу отвечать только про этот товар."
            response = self._build_response(
                conversation_id=conversation_id,
                reply_text=reply_text,
                confidence=0.1,
                used_fields=[],
                context_hash=context_hash,
                context_cache_hit=context_cache_hit,
                out_of_scope=policy_decision.out_of_scope,
                refusal_reason=policy_decision.refusal_reason,
                injection_detected=policy_decision.injection_detected,
                llm_used=False,
            )
            self._record_history(conversation_id, request.message, response.reply.text)
            return response

        # Check answer cache FIRST (before rate limit) to avoid 429 for cached answers
        cached_answer = self._cache.get_product_chat_answer(
            product_id=request.product_id,
            question=request.message,
            context_hash=context_hash,
        )
        if cached_answer:
            logger.info(
                "product_chat.cache_hit conversation_id=%s question=%s",
                conversation_id,
                request.message[:50],
            )
            response = self._build_response(
                conversation_id=conversation_id,
                reply_text=cached_answer["reply_text"],
                confidence=cached_answer["confidence"],
                used_fields=cached_answer.get("used_fields", []),
                context_hash=context_hash,
                context_cache_hit=context_cache_hit,
                refusal_reason=cached_answer.get("refusal_reason"),
                llm_used=False,  # From cache
                answer_cache_hit=True,
            )
            self._record_history(conversation_id, request.message, response.reply.text)
            return response

        # Rate limit check only when we need to call LLM
        if not self._metrics.check_rate_limit(
            user_id=request.user_id,
            window_seconds=self._settings.llm_rate_limit_window_seconds,
            max_calls=self._settings.llm_rate_limit_max_calls,
        ):
            # Audit: log rate limit
            self._audit.log_rate_limit(
                trace_id=trace_id or conversation_id,
                user_id=request.user_id,
                product_id=request.product_id,
            )
            raise BadRequestError(
                "Превышен лимит запросов. Пожалуйста, подождите немного.",
                reason="rate_limit_exceeded",
                http_status=429,
            )

        if self._llm_client is None:
            response = self._build_response(
                conversation_id=conversation_id,
                reply_text="Сервис временно недоступен. Попробуйте позже.",
                confidence=0.0,
                used_fields=[],
                context_hash=context_hash,
                context_cache_hit=context_cache_hit,
                refusal_reason=ProductChatRefusalReason.NO_DATA,
                llm_used=False,
            )
            self._record_history(conversation_id, request.message, response.reply.text)
            return response

        history_text = self._format_history(conversation_id)
        llm_run = await self._llm_client.answer_product_question(
            product_id=request.product_id,
            message=request.message,
            context_json=context,
            conversation_history=history_text,
            conversation_id=conversation_id,
            user_id=request.user_id,
            trace_id=trace_id,
        )
        llm_result = llm_run.result

        reply_text = llm_result.answer.strip()
        if llm_result.needs_clarification and llm_result.clarifying_question:
            if reply_text and llm_result.clarifying_question not in reply_text:
                reply_text = f"{reply_text} {llm_result.clarifying_question}".strip()
            elif not reply_text:
                reply_text = llm_result.clarifying_question

        used_fields_raw = list(llm_result.used_fields or [])
        used_fields, invalid_used_fields = self._filter_used_fields(context, used_fields_raw)
        refusal_reason = llm_result.refusal_reason
        out_of_scope = bool(llm_result.out_of_scope)

        if refusal_reason or out_of_scope:
            used_fields = []
        else:
            if self._policy_guard.is_dosage_request(request.message) and not self._context_has_dosage(context):
                refusal_reason = ProductChatRefusalReason.NO_DATA
                if self._context_has_instructions(context):
                    reply_text = (
                        "В инструкции к товару есть сведения о способе применения. "
                        "Ознакомьтесь с документом на странице товара."
                    )
                else:
                    reply_text = (
                        "В карточке товара нет данных о способе применения и дозировках. "
                        "Уточните, что именно вас интересует."
                    )
                used_fields = []
            # NOTE: Removed overly strict validation that blocked responses when
            # LLM returned invalid field paths. LLM may return meaningful answers
            # even if used_fields paths don't exactly match context structure.

        response = self._build_response(
            conversation_id=conversation_id,
            reply_text=reply_text or "Готово.",
            confidence=llm_result.confidence,
            used_fields=used_fields,
            context_hash=context_hash,
            context_cache_hit=context_cache_hit,
            out_of_scope=out_of_scope,
            refusal_reason=refusal_reason,
            injection_detected=False,
            llm_used=True,
            llm_cached=llm_run.cached,
            token_usage=llm_run.token_usage,
            invalid_used_fields=invalid_used_fields,
        )
        self._record_history(conversation_id, request.message, response.reply.text)
        
        # Cache the answer for future similar questions (only if successful)
        if not out_of_scope and not refusal_reason:
            self._cache.set_product_chat_answer(
                product_id=request.product_id,
                question=request.message,
                context_hash=context_hash,
                answer_payload={
                    "reply_text": reply_text or "Готово.",
                    "confidence": llm_result.confidence,
                    "used_fields": used_fields,
                    "refusal_reason": None,
                },
                ttl_seconds=300,  # 5 minutes
            )
            logger.debug(
                "product_chat.cache_set product_id=%s question=%s",
                request.product_id,
                request.message[:50],
            )
        
        # Audit: log successful response
        token_usage = llm_run.token_usage or {}
        self._audit.log_chat_response(
            trace_id=trace_id or conversation_id,
            conversation_id=conversation_id,
            user_id=request.user_id,
            product_id=request.product_id,
            context_hash=context_hash,
            context_cache_hit=context_cache_hit,
            context_sources=["product-search"],
            llm_provider=self._settings.llm_provider,
            llm_model=self._settings.yandex_model if self._settings.llm_provider == "yandex" else self._settings.openai_model,
            llm_tokens_input=token_usage.get("prompt_tokens", 0),
            llm_tokens_output=token_usage.get("completion_tokens", 0),
            llm_cached=llm_run.cached,
            success=True,
            response_text=reply_text,
            used_fields=used_fields,
            confidence=llm_result.confidence,
            blocked=False,
            refusal_reason=refusal_reason.value if refusal_reason else None,
            out_of_scope=out_of_scope,
        )
        
        return response

    def _record_history(self, conversation_id: str, user_message: str, assistant_text: str) -> None:
        self._conversation_store.append_message(conversation_id, "user", user_message)
        if assistant_text:
            self._conversation_store.append_message(conversation_id, "assistant", assistant_text)

    def _format_history(self, conversation_id: str) -> str:
        history = self._conversation_store.get_history(conversation_id, limit=10)
        parts: List[str] = []
        for item in history:
            role = "Пользователь" if item.role == "user" else "Ассистент"
            parts.append(f"{role}: {item.content}")
        return "\n".join(parts)

    def _filter_used_fields(
        self, context: Dict[str, Any], used_fields: List[str]
    ) -> tuple[List[str], List[str]]:
        valid: List[str] = []
        invalid: List[str] = []
        for field in used_fields:
            normalized = (field or "").strip()
            if not normalized or normalized in valid:
                continue
            if self._path_exists(context, normalized):
                valid.append(normalized)
            else:
                invalid.append(normalized)
        return valid, invalid

    def _path_exists(self, context: Dict[str, Any], path: str) -> bool:
        current: Any = context
        for raw_part in path.split("."):
            if not raw_part:
                continue
            name, indexes = self._split_part(raw_part)
            if name:
                if isinstance(current, dict):
                    if name not in current:
                        return False
                    current = current[name]
                elif isinstance(current, list):
                    value_found = None
                    for item in current:
                        if isinstance(item, dict) and name in item:
                            value_found = item[name]
                            break
                    if value_found is None:
                        return False
                    current = value_found
                else:
                    return False
            for idx in indexes:
                if isinstance(current, list) and 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return False
        return True

    @classmethod
    def _split_part(cls, part: str) -> tuple[str, List[int]]:
        if "[" not in part:
            return part, []
        name = part.split("[", 1)[0]
        indexes = [int(match) for match in cls._PATH_INDEX_RE.findall(part)]
        return name, indexes

    @staticmethod
    def _context_has_dosage(context: Dict[str, Any]) -> bool:
        pharma_info = context.get("pharma_info") or {}
        if pharma_info.get("dosage"):
            return True
        attributes = context.get("attributes") or []
        for attr in attributes:
            code = (attr.get("code") or "").lower()
            if code in {"dosage", "directions"} and attr.get("value"):
                return True
        return False

    @staticmethod
    def _context_has_instructions(context: Dict[str, Any]) -> bool:
        documents = context.get("documents") or {}
        instructions = documents.get("instructions") or []
        return bool(instructions)

    def _build_response(
        self,
        *,
        conversation_id: str,
        reply_text: str,
        confidence: float,
        used_fields: List[str],
        context_hash: str,
        context_cache_hit: bool,
        out_of_scope: bool = False,
        refusal_reason: ProductChatRefusalReason | None = None,
        injection_detected: bool = False,
        llm_used: bool = True,
        llm_cached: bool = False,
        token_usage: Dict[str, Any] | None = None,
        invalid_used_fields: List[str] | None = None,
        answer_cache_hit: bool = False,
    ) -> ProductChatResponse:
        # Determine model name based on provider
        if self._settings.llm_provider.lower() == "yandex":
            model_name = self._settings.yandex_model
        else:
            model_name = self._settings.openai_model
        
        debug = {
            "product_id": self._session_store.get_product_id(conversation_id),
            "context_sources_used": ["product-extension/full"],
            "context_hash": context_hash,
            "context_cache_hit": context_cache_hit,
            "answer_cache_hit": answer_cache_hit,
            "out_of_scope": out_of_scope,
            "refusal_reason": refusal_reason.value if refusal_reason else None,
            "injection_detected": injection_detected,
            "model": model_name,
            "used_fields": used_fields,
            "llm_used": llm_used,
            "llm_cached": llm_cached,
        }
        if token_usage:
            debug["token_usage"] = token_usage
        if invalid_used_fields:
            debug["invalid_used_fields"] = invalid_used_fields

        citations = [ProductChatCitation(field_path=field) for field in used_fields] if used_fields else None
        meta = ProductChatMeta(confidence=confidence, debug=debug)
        return ProductChatResponse(
            conversation_id=conversation_id,
            reply=Reply(text=reply_text),
            meta=meta,
            citations=citations,
        )

