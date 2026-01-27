from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from ..models.product_chat import ProductChatRefusalReason


@dataclass
class ProductPolicyDecision:
    blocked: bool
    out_of_scope: bool = False
    refusal_reason: ProductChatRefusalReason | None = None
    injection_detected: bool = False
    reply_text: str | None = None
    is_cyclic: bool = False  # Новое поле для циклических запросов


class ProductPolicyGuard:
    """
    Policy Guard для Product AI Chat.
    
    Проверяет:
    - Prompt injection
    - Out of scope вопросы
    - Фарма-ограничения (дозировки для рецептурных)
    - Циклические/повторяющиеся вопросы
    """
    
    _INJECTION_MARKERS = (
        "system prompt",
        "developer message",
        "ignore previous",
        "раскрой инструкции",
        "покажи контекст",
        "покажи промпт",
        "скрой правила",
        "chain of thought",
        "prompt injection",
    )

    _OUT_OF_SCOPE_MARKERS = (
        "что лучше вообще",
        "какой лучше вообще",
        "какой телефон лучше",
        "посоветуй другой",
        "посоветуй что купить",
        "любой товар",
        "лучший товар",
    )

    _DOSAGE_MARKERS = (
        "дозиров",
        "сколько таблет",
        "сколько кап",
        "по сколько",
        "сколько раз в день",
        "курс",
        "схема приема",
        "как принимать",
        "как пить",
    )

    # Настройки защиты от циклических запросов
    CYCLIC_WINDOW_SECONDS = 300  # 5 минут
    CYCLIC_MAX_SIMILAR = 3  # Максимум похожих запросов
    CYCLIC_SIMILARITY_THRESHOLD = 0.8  # Порог схожести
    
    def __init__(self):
        # Хранение хешей сообщений: {conversation_id: [(timestamp, message_hash), ...]}
        self._message_history: Dict[str, List[tuple]] = defaultdict(list)
    
    def _normalize_message(self, message: str) -> str:
        """Нормализация сообщения для сравнения."""
        # Удаляем знаки препинания, приводим к нижнему регистру
        normalized = message.lower().strip()
        # Удаляем лишние пробелы
        normalized = " ".join(normalized.split())
        return normalized
    
    def _message_hash(self, message: str) -> str:
        """Хеш нормализованного сообщения."""
        normalized = self._normalize_message(message)
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _is_similar_message(self, msg1: str, msg2_hash: str) -> bool:
        """Проверяет похожесть сообщений по хешу."""
        return self._message_hash(msg1) == msg2_hash
    
    def _check_cyclic(
        self, message: str, conversation_id: Optional[str]
    ) -> bool:
        """
        Проверяет, является ли сообщение циклическим/повторяющимся.
        
        Returns:
            True если обнаружен цикл
        """
        if not conversation_id:
            return False
        
        current_time = time.time()
        message_hash = self._message_hash(message)
        
        # Очищаем старые записи
        self._message_history[conversation_id] = [
            (ts, h) for ts, h in self._message_history[conversation_id]
            if current_time - ts < self.CYCLIC_WINDOW_SECONDS
        ]
        
        # Считаем похожие сообщения
        similar_count = sum(
            1 for _, h in self._message_history[conversation_id]
            if h == message_hash
        )
        
        # Добавляем текущее сообщение
        self._message_history[conversation_id].append((current_time, message_hash))
        
        # Ограничиваем размер истории
        if len(self._message_history[conversation_id]) > 50:
            self._message_history[conversation_id] = self._message_history[conversation_id][-50:]
        
        return similar_count >= self.CYCLIC_MAX_SIMILAR
    
    def evaluate(
        self,
        message: str,
        context: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> ProductPolicyDecision:
        normalized = (message or "").strip().lower()
        if not normalized:
            return ProductPolicyDecision(blocked=True, refusal_reason=ProductChatRefusalReason.NO_DATA, reply_text="Уточните вопрос о товаре.")

        # 1. Prompt injection check
        if any(marker in normalized for marker in self._INJECTION_MARKERS):
            return ProductPolicyDecision(
                blocked=True,
                out_of_scope=True,
                refusal_reason=ProductChatRefusalReason.PROMPT_INJECTION,
                injection_detected=True,
                reply_text="Я не могу помочь с этим. Могу отвечать только по параметрам товара.",
            )

        # 2. Out of scope check
        if any(marker in normalized for marker in self._OUT_OF_SCOPE_MARKERS):
            return ProductPolicyDecision(
                blocked=True,
                out_of_scope=True,
                refusal_reason=ProductChatRefusalReason.OUT_OF_SCOPE,
                reply_text="Я могу отвечать только про этот товар. Спросите про характеристики, цену, наличие или доставку.",
            )

        # 3. Pharma dosage restrictions
        prescription = bool((context.get("product") or {}).get("prescription"))
        if prescription and any(marker in normalized for marker in self._DOSAGE_MARKERS):
            return ProductPolicyDecision(
                blocked=True,
                out_of_scope=False,
                refusal_reason=ProductChatRefusalReason.POLICY_RESTRICTED,
                reply_text="Я не могу назначать дозировки. Следуйте инструкции или обратитесь к врачу.",
            )

        # 4. Cyclic/repetitive questions check
        if self._check_cyclic(message, conversation_id):
            return ProductPolicyDecision(
                blocked=True,
                out_of_scope=False,
                refusal_reason=ProductChatRefusalReason.POLICY_RESTRICTED,
                is_cyclic=True,
                reply_text="Вы уже задавали похожий вопрос. Попробуйте переформулировать или спросите о другом аспекте товара.",
            )

        return ProductPolicyDecision(blocked=False)

