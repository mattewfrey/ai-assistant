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


_SHOW_VERBS = (
    "покажи",
    "открой",
    "посмотри",
    "что в",
    "перейти в",
    "отобрази",
)

_ACTION_GUARDS = (
    "добав",
    "полож",
    "закин",
    "убер",
    "удал",
    "очист",
    "измени",
    "примен",
    "активир",
    "обнов",
    "поменя",
    "купить",
    "заказать",
)

_CART_PATTERNS = (
    r"(?:покажи|открой|посмотри)\s+корзин",
    r"что в корзин",
    r"перейти в корзин",
    r"\bкорзина$",
    r"\bкорзину$",
)
_ORDER_HISTORY_PATTERNS = (r"мои заказы", r"истори[яи] заказ", r"последние заказы")
_ACTIVE_ORDERS_HINT = r"активн"
_FAVORITES_PATTERNS = (
    r"(?:покажи|открой|посмотри)\s+избран",
    r"избранные товары",
    r"\bизбранное$",
)
_PROFILE_PATTERNS = (
    r"(?:покажи|открой|посмотри)\s+профил",
    r"\bмой профиль\b",
    r"\bпрофиль\b",
)
_NAVIGATION_GUARDS = ("корзин", "заказ", "избран", "профил")

# Фразы, которые НЕ являются запросами товаров (приветствия, благодарности и т.д.)
_SMALL_TALK_PATTERNS = (
    r"^\s*привет\s*$",
    r"^\s*здравствуй",
    r"^\s*добр(ый|ое|ого)\s+(день|утр|вечер)",
    r"^\s*хай\s*$",
    r"^\s*hi\s*$",
    r"^\s*hello\s*$",
    r"^\s*спасибо\s*$",
    r"^\s*благодар",
    r"^\s*пока\s*$",
    r"^\s*до\s+свидания",
    r"^\s*давай\s*$",
    r"^\s*ок(ей)?\s*$",
    r"^\s*хорошо\s*$",
    r"^\s*понятно\s*$",
    r"^\s*ясно\s*$",
    r"^\s*да\s*$",
    r"^\s*нет\s*$",
    r"^\s*ага\s*$",
    r"^\s*угу\s*$",
    r"^\s*ладно\s*$",
    r"^\s*что\??\s*$",
    r"^\s*как дела",
    r"^\s*что нового",
    r"^\s*что умеешь",
    r"^\s*помо(щь|ги|жешь)",
)


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _is_small_talk(text: str) -> bool:
    """Check if text is a greeting, thank you, or other non-product phrase."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in _SMALL_TALK_PATTERNS)


def _looks_like_product_name(text: str) -> bool:
    """Check if text looks like a product name (has uppercase, digits, or brand-like patterns)."""
    # Содержит цифры (дозировка: "500мг", "100мл")
    if re.search(r"\d", text):
        return True
    # Начинается с заглавной буквы (название бренда)
    if text[0].isupper():
        return True
    # Содержит латиницу (международное название)
    if re.search(r"[a-zA-Z]", text):
        return True
    return False


def _is_short_product_query(text: str) -> bool:
    """Check if text is a short query that looks like a product search."""
    tokens = [token for token in re.split(r"\s+", text.strip()) if token]
    if not (1 <= len(tokens) <= 3):
        return False
    
    # Отклоняем short talk фразы
    if _is_small_talk(text):
        return False
    
    # Требуем признаки продукта: цифры, заглавные буквы или латиницу
    return _looks_like_product_name(text)


def route(request: ChatRequest) -> LocalRouterResult:
    """
    Attempt to resolve very simple, deterministic intents without calling the LLM.
    """

    text = (request.message or "").strip()
    if not text:
        return LocalRouterResult(matched=False)

    normalized = text.lower()

    if _contains_any(normalized, _ACTION_GUARDS):
        return LocalRouterResult(matched=False)

    if _contains_any(normalized, _CART_PATTERNS) and _contains_any(normalized, _SHOW_VERBS + ("что в",)):
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


