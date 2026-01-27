from __future__ import annotations

import logging
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
from .conversation_store import ConversationStore, get_conversation_store
from .metrics import MetricsService, get_metrics_service
from .product_chat_session_store import ProductChatSessionStore, get_product_chat_session_store
from .product_context_builder import ProductContextBuilder
from .product_policy_guard import ProductPolicyGuard
from .errors import BadRequestError

logger = logging.getLogger(__name__)


class ProductChatService:
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
    ) -> None:
        self._settings = settings
        self._context_builder = context_builder
        self._policy_guard = policy_guard
        self._llm_client = llm_client
        self._session_store = session_store or get_product_chat_session_store()
        self._conversation_store = conversation_store or get_conversation_store()
        self._metrics = metrics or get_metrics_service()
        self._audit = audit or get_audit_service()

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

        used_fields = list(llm_result.used_fields or [])
        refusal_reason = llm_result.refusal_reason
        out_of_scope = bool(llm_result.out_of_scope)

        if not used_fields and not refusal_reason and not out_of_scope:
            refusal_reason = ProductChatRefusalReason.NO_DATA
            reply_text = "В карточке товара этого не указано. Уточните, что именно вас интересует."

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
        )
        self._record_history(conversation_id, request.message, response.reply.text)
        
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

        citations = [ProductChatCitation(field_path=field) for field in used_fields] if used_fields else None
        meta = ProductChatMeta(confidence=confidence, debug=debug)
        return ProductChatResponse(
            conversation_id=conversation_id,
            reply=Reply(text=reply_text),
            meta=meta,
            citations=citations,
        )

