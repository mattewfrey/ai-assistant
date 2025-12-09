from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

from pydantic import BaseModel, Field

from ..intents import IntentType
from ..models import ChatRequest


class LocalRouterResult(BaseModel):
    """Lightweight, rule-based routing outcome."""

    matched: bool
    intent: Optional[IntentType] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


_CART_PATTERNS = (r"\bкорзин", r"что в корзине")
_ORDER_HISTORY_PATTERNS = (r"мои заказы", r"истори[яи] заказ", r"последние заказы")
_ACTIVE_ORDERS_HINT = r"активн"
_FAVORITES_PATTERNS = (r"избранн", r"избранные товары")
_PROFILE_PATTERNS = (r"мой профиль", r"\bпрофиль\b")
_NAVIGATION_GUARDS = ("корзин", "заказ", "избран", "профил")


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _is_short_product_query(text: str) -> bool:
    tokens = [token for token in re.split(r"\s+", text.strip()) if token]
    return 1 <= len(tokens) <= 3


def route(request: ChatRequest) -> LocalRouterResult:
    """
    Attempt to resolve very simple, deterministic intents without calling the LLM.
    """

    text = (request.message or "").strip()
    if not text:
        return LocalRouterResult(matched=False)

    normalized = text.lower()

    if _contains_any(normalized, _CART_PATTERNS):
        return LocalRouterResult(
            matched=True,
            intent=IntentType.SHOW_CART,
            parameters={},
        )

    if _contains_any(normalized, _ORDER_HISTORY_PATTERNS):
        intent = IntentType.SHOW_ACTIVE_ORDERS if re.search(_ACTIVE_ORDERS_HINT, normalized) else IntentType.SHOW_ORDER_HISTORY
        return LocalRouterResult(matched=True, intent=intent, parameters={})

    if _contains_any(normalized, _FAVORITES_PATTERNS):
        return LocalRouterResult(matched=True, intent=IntentType.SHOW_FAVORITES, parameters={})

    if _contains_any(normalized, _PROFILE_PATTERNS):
        return LocalRouterResult(matched=True, intent=IntentType.SHOW_PROFILE, parameters={})

    if _is_short_product_query(text) and not _contains_any(normalized, _NAVIGATION_GUARDS):
        clean_text = text.strip()
        return LocalRouterResult(
            matched=True,
            intent=IntentType.FIND_PRODUCT_BY_NAME,
            parameters={"product_name": clean_text, "name": clean_text},
        )

    return LocalRouterResult(matched=False)


