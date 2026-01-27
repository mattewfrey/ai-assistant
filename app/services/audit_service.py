"""
Audit Service — структурированное логирование для Product AI Chat.

Записывает все запросы к LLM с полным контекстом для анализа и отладки.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Типы событий аудита."""
    CHAT_REQUEST = "chat_request"
    CHAT_RESPONSE = "chat_response"
    POLICY_BLOCK = "policy_block"
    LLM_CALL = "llm_call"
    RATE_LIMIT = "rate_limit"
    ERROR = "error"


class AuditChannel(str, Enum):
    """Каналы использования."""
    STOREFRONT = "storefront"
    SUPPORT = "support"
    API = "api"
    DEMO = "demo"


@dataclass
class AuditEvent:
    """Структура события аудита."""
    
    # Идентификаторы
    event_type: AuditEventType
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    trace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Кто
    user_id: Optional[str] = None
    channel: AuditChannel = AuditChannel.API
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Что
    product_id: Optional[str] = None
    message: Optional[str] = None
    message_length: int = 0
    
    # Контекст
    context_hash: Optional[str] = None
    context_cache_hit: bool = False
    context_sources: List[str] = field(default_factory=list)
    
    # LLM
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_tokens_input: int = 0
    llm_tokens_output: int = 0
    llm_tokens_total: int = 0
    llm_cached: bool = False
    llm_latency_ms: int = 0
    
    # Результат
    success: bool = True
    response_text: Optional[str] = None
    response_length: int = 0
    used_fields: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    # Отказы
    blocked: bool = False
    refusal_reason: Optional[str] = None
    out_of_scope: bool = False
    injection_detected: bool = False
    
    # Ошибки
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert enums to strings
        data["event_type"] = self.event_type.value
        data["channel"] = self.channel.value
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AuditService:
    """
    Сервис аудита для записи всех событий Product AI Chat.
    
    Записывает в структурированном формате для последующего анализа.
    В production можно расширить для отправки в ELK, ClickHouse и т.д.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        self._logger = logging.getLogger("audit.product_chat")
        self._log_level = log_level
        
        # Ensure audit logger is configured
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s AUDIT %(message)s'
            ))
            self._logger.addHandler(handler)
            self._logger.setLevel(log_level)
    
    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        self._logger.log(self._log_level, event.to_json())
    
    def log_chat_request(
        self,
        *,
        trace_id: str,
        conversation_id: str,
        user_id: Optional[str],
        product_id: str,
        message: str,
        channel: AuditChannel = AuditChannel.API,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        """Log incoming chat request."""
        event = AuditEvent(
            event_type=AuditEventType.CHAT_REQUEST,
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
            product_id=product_id,
            message=message[:200] if message else None,  # Truncate for logs
            message_length=len(message) if message else 0,
            channel=channel,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        self.log_event(event)
    
    def log_chat_response(
        self,
        *,
        trace_id: str,
        conversation_id: str,
        user_id: Optional[str],
        product_id: str,
        # Context
        context_hash: str,
        context_cache_hit: bool,
        context_sources: List[str],
        # LLM
        llm_provider: str,
        llm_model: str,
        llm_tokens_input: int = 0,
        llm_tokens_output: int = 0,
        llm_cached: bool = False,
        llm_latency_ms: int = 0,
        # Result
        success: bool,
        response_text: Optional[str],
        used_fields: List[str],
        confidence: float,
        # Refusals
        blocked: bool = False,
        refusal_reason: Optional[str] = None,
        out_of_scope: bool = False,
        injection_detected: bool = False,
    ) -> None:
        """Log chat response with full details."""
        event = AuditEvent(
            event_type=AuditEventType.CHAT_RESPONSE,
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
            product_id=product_id,
            # Context
            context_hash=context_hash,
            context_cache_hit=context_cache_hit,
            context_sources=context_sources,
            # LLM
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_tokens_input=llm_tokens_input,
            llm_tokens_output=llm_tokens_output,
            llm_tokens_total=llm_tokens_input + llm_tokens_output,
            llm_cached=llm_cached,
            llm_latency_ms=llm_latency_ms,
            # Result
            success=success,
            response_text=response_text[:200] if response_text else None,
            response_length=len(response_text) if response_text else 0,
            used_fields=used_fields,
            confidence=confidence,
            # Refusals
            blocked=blocked,
            refusal_reason=refusal_reason,
            out_of_scope=out_of_scope,
            injection_detected=injection_detected,
        )
        self.log_event(event)
    
    def log_policy_block(
        self,
        *,
        trace_id: str,
        conversation_id: str,
        user_id: Optional[str],
        product_id: str,
        message: str,
        refusal_reason: str,
        out_of_scope: bool = False,
        injection_detected: bool = False,
    ) -> None:
        """Log when request is blocked by policy."""
        event = AuditEvent(
            event_type=AuditEventType.POLICY_BLOCK,
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
            product_id=product_id,
            message=message[:200] if message else None,
            blocked=True,
            refusal_reason=refusal_reason,
            out_of_scope=out_of_scope,
            injection_detected=injection_detected,
            success=False,
        )
        self.log_event(event)
    
    def log_rate_limit(
        self,
        *,
        trace_id: str,
        user_id: Optional[str],
        product_id: str,
        ip_address: Optional[str] = None,
    ) -> None:
        """Log rate limit hit."""
        event = AuditEvent(
            event_type=AuditEventType.RATE_LIMIT,
            trace_id=trace_id,
            user_id=user_id,
            product_id=product_id,
            ip_address=ip_address,
            blocked=True,
            refusal_reason="RATE_LIMIT",
            success=False,
        )
        self.log_event(event)
    
    def log_error(
        self,
        *,
        trace_id: str,
        conversation_id: Optional[str],
        user_id: Optional[str],
        product_id: Optional[str],
        error_code: str,
        error_message: str,
    ) -> None:
        """Log error event."""
        event = AuditEvent(
            event_type=AuditEventType.ERROR,
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
            product_id=product_id,
            error_code=error_code,
            error_message=error_message[:500] if error_message else None,
            success=False,
        )
        self.log_event(event)


# Singleton instance
_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """Get or create singleton audit service."""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service
