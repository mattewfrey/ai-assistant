"""
LLM Debug Store - хранилище детальной информации о вызовах LLM.

Позволяет просматривать:
- Полные промпты, отправленные в LLM
- Полные ответы от LLM
- Токены, время выполнения
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class LLMCallRecord:
    """Запись об одном вызове LLM."""
    
    timestamp: str
    call_type: str  # "classify_intent", "product_chat", "beautify", "disambiguate", etc.
    model: str
    
    # Входные данные
    user_message: str
    system_prompt: str
    full_prompt: str  # Полный промпт (если доступен)
    
    # Выходные данные
    raw_response: str  # Сырой ответ от LLM
    parsed_response: Dict[str, Any]  # Распарсенный ответ
    
    # Метаданные
    token_usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    cached: bool = False
    error: Optional[str] = None
    
    # Контекст
    trace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    product_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "call_type": self.call_type,
            "model": self.model,
            "user_message": self.user_message,
            "system_prompt_preview": self.system_prompt[:500] + "..." if len(self.system_prompt) > 500 else self.system_prompt,
            "system_prompt_full": self.system_prompt,
            "full_prompt": self.full_prompt,
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
            "token_usage": self.token_usage,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "error": self.error,
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "product_id": self.product_id,
        }


class LLMDebugStore:
    """Thread-safe хранилище для LLM вызовов."""
    
    _instance: Optional["LLMDebugStore"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "LLMDebugStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, max_records: int = 50) -> None:
        if self._initialized:
            return
        self._records: deque[LLMCallRecord] = deque(maxlen=max_records)
        self._records_lock = threading.Lock()
        self._initialized = True
    
    def record_call(
        self,
        *,
        call_type: str,
        model: str,
        user_message: str,
        system_prompt: str = "",
        full_prompt: str = "",
        raw_response: str = "",
        parsed_response: Dict[str, Any] | None = None,
        token_usage: Dict[str, int] | None = None,
        latency_ms: float = 0.0,
        cached: bool = False,
        error: str | None = None,
        trace_id: str | None = None,
        conversation_id: str | None = None,
        product_id: str | None = None,
    ) -> None:
        """Записывает информацию о вызове LLM."""
        record = LLMCallRecord(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            call_type=call_type,
            model=model,
            user_message=user_message,
            system_prompt=system_prompt,
            full_prompt=full_prompt,
            raw_response=raw_response,
            parsed_response=parsed_response or {},
            token_usage=token_usage or {},
            latency_ms=latency_ms,
            cached=cached,
            error=error,
            trace_id=trace_id,
            conversation_id=conversation_id,
            product_id=product_id,
        )
        
        with self._records_lock:
            self._records.append(record)
    
    def get_records(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Возвращает последние записи."""
        with self._records_lock:
            records = list(self._records)
        return [r.to_dict() for r in records[-limit:]]
    
    def get_last_record(self) -> Optional[Dict[str, Any]]:
        """Возвращает последнюю запись."""
        with self._records_lock:
            if self._records:
                return self._records[-1].to_dict()
        return None
    
    def clear(self) -> None:
        """Очищает все записи."""
        with self._records_lock:
            self._records.clear()


def get_llm_debug_store() -> LLMDebugStore:
    """Возвращает singleton экземпляр LLMDebugStore."""
    return LLMDebugStore()
