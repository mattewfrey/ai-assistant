"""Response building helpers extracted from orchestrator for reusability."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..intents import IntentType
from ..models import AssistantMeta

# ============================================================================
# Fallback reply text constants
# ============================================================================

UNKNOWN_FALLBACK_REPLY = "Я пока не понял запрос. Могу помочь с подбором препаратов, заказами или корзиной."
LOW_CONFIDENCE_FALLBACK = "Кажется, я не до конца понял запрос. Можете уточнить, что именно нужно?"
SMALL_TALK_FALLBACK_REPLY = "Всегда рад помочь! Если потребуется что-то из аптеки, просто скажите."

# ============================================================================
# Quick reply presets
# ============================================================================

DEFAULT_QUICK_REPLY_PRESETS: List[Dict[str, Any]] = [
    {
        "label": "Подбор по симптомам",
        "query": "Подбери препарат по моим симптомам",
        "intent": IntentType.FIND_BY_SYMPTOM.value,
        "parameters": {},
    },
    {
        "label": "Помощь по заказу",
        "query": "Покажи статус моего заказа",
        "intent": IntentType.SHOW_ORDER_STATUS.value,
        "parameters": {},
    },
    {
        "label": "Показать корзину",
        "query": "Покажи мою корзину",
        "intent": IntentType.SHOW_CART.value,
        "parameters": {},
    },
    {
        "label": "Выбрать аптеку",
        "query": "Покажи ближайшие аптеки",
        "intent": IntentType.SHOW_NEARBY_PHARMACIES.value,
        "parameters": {},
    },
]


def default_quick_replies() -> List[Dict[str, Any]]:
    """Return a copy of default quick reply presets."""
    return [preset.copy() for preset in DEFAULT_QUICK_REPLY_PRESETS]


def symptom_clarification_quick_replies(intent: IntentType | None) -> List[Dict[str, Any]]:
    """Generate quick replies for age clarification in symptom queries."""
    if not intent:
        return []
    intent_value = intent.value
    return [
        {
            "label": "Ребёнок до 6 лет",
            "query": "Нам 5 лет",
            "intent": intent_value,
            "parameters": {"age": 5},
        },
        {
            "label": "Подросток 12-17",
            "query": "Возраст 14 лет",
            "intent": intent_value,
            "parameters": {"age": 14},
        },
        {
            "label": "Взрослый 18+",
            "query": "Мне 30 лет",
            "intent": intent_value,
            "parameters": {"age": 30},
        },
    ]


def no_results_quick_replies() -> List[Dict[str, Any]]:
    """Generate quick replies when product search returned no results."""
    return [
        {
            "label": "Показать популярные категории",
            "query": "Покажи популярные категории",
            "intent": IntentType.FIND_BY_CATEGORY.value,
            "parameters": {"category": "popular"},
        },
        {
            "label": "Подобрать без фильтров",
            "query": "Подбери препараты заново",
            "intent": IntentType.FIND_BY_SYMPTOM.value,
            "parameters": {},
        },
        {
            "label": "Сменить аптеку",
            "query": "Покажи другие аптеки",
            "intent": IntentType.SHOW_NEARBY_PHARMACIES.value,
            "parameters": {},
        },
    ]


def merge_quick_replies(
    existing: List[Dict[str, Any]] | None,
    additions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge quick reply lists, putting additions first."""
    if not additions:
        return existing or []
    merged = [reply.copy() for reply in additions]
    if existing:
        merged.extend(existing)
    return merged


# ============================================================================
# Symptom parameter helpers
# ============================================================================

SYMPTOM_PARAM_ALIASES: Dict[str, Sequence[str]] = {
    "age": ("age", "age_years", "age_group"),
}

SYMPTOM_FIELD_QUESTIONS: Dict[str, str] = {
    "age": "Сколько лет человеку, для которого подбираем лечение?",
}

SYMPTOM_EXTRA_HINT = (
    "Также уточните температуру, длительность симптомов и хронические заболевания, "
    "чтобы подобрать безопасные товары."
)


def has_non_empty_value(parameters: Dict[str, Any], keys: Sequence[str]) -> bool:
    """Check if any of the keys has a non-empty value in parameters."""
    for key in keys:
        if key not in parameters:
            continue
        value = parameters[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return True
    return False


def missing_symptom_parameters(parameters: Dict[str, Any]) -> List[str]:
    """Return list of missing required symptom parameters."""
    missing: List[str] = []
    if not has_non_empty_value(parameters, SYMPTOM_PARAM_ALIASES["age"]):
        missing.append("age")
    return missing


def build_symptom_clarification_text(
    previous_text: str | None,
    missing_fields: Sequence[str],
) -> str:
    """Build clarification text for missing symptom parameters."""
    parts: List[str] = []
    if previous_text:
        stripped = previous_text.strip()
        if stripped:
            parts.append(stripped)
    for field in missing_fields:
        question = SYMPTOM_FIELD_QUESTIONS.get(field)
        if question:
            parts.append(question)
    parts.append(SYMPTOM_EXTRA_HINT)
    return " ".join(part.strip() for part in parts if part).strip()


def no_results_reply_text(previous_text: str | None) -> str:
    """Build reply text when no products were found."""
    base = "Пока не нашёл подходящих товаров с учётом выбранной аптеки и фильтров."
    suffix = " Могу показать популярные категории или подобрать альтернативы."
    if previous_text:
        return f"{base} {previous_text.strip()}"
    return f"{base}{suffix}"


# ============================================================================
# Product helpers
# ============================================================================

PRODUCT_RESULT_LIMIT = 8


def deduplicate_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate products by ID."""
    seen_ids: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for product in products:
        product_id = product.get("id")
        if product_id and product_id in seen_ids:
            continue
        if product_id:
            seen_ids.add(product_id)
        unique.append(product)
    return unique


# ============================================================================
# Purchase history helpers  
# ============================================================================


def build_purchase_history_quick_replies(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build quick replies based on purchase history."""
    if not orders:
        return []
    
    quick_replies: List[Dict[str, Any]] = []
    last_order = orders[0]
    last_items = last_order.get("items") or []
    
    if last_items:
        quick_replies.append({
            "label": "Повторить последний заказ",
            "query": "Повтори мой последний заказ",
            "intent": None,
            "parameters": {"order_id": last_order.get("order_id")},
        })
    
    frequent_item = most_frequent_item(orders)
    if frequent_item:
        item_name = frequent_item.get("title") or "ваши частые покупки"
        quick_replies.append({
            "label": "Показать частые покупки",
            "query": "Показать товары, которые я часто покупаю",
            "intent": IntentType.FIND_PRODUCT_BY_NAME.value,
            "parameters": {
                "product_id": frequent_item.get("product_id"),
                "name": item_name,
            },
        })
    
    return quick_replies


def most_frequent_item(orders: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Find the most frequently purchased item across orders."""
    stats: Dict[str, Dict[str, Any]] = {}
    
    for order in orders:
        for item in order.get("items") or []:
            product_id = item.get("product_id")
            if not product_id:
                continue
            bucket = stats.setdefault(product_id, {"count": 0, "title": item.get("title")})
            bucket["count"] += 1
            if not bucket.get("title") and item.get("title"):
                bucket["title"] = item["title"]
    
    if not stats:
        return None
    
    product_id, aggregates = max(stats.items(), key=lambda entry: entry[1]["count"])
    if aggregates["count"] < 2:
        return None
    
    return {"product_id": product_id, "title": aggregates.get("title")}


__all__ = [
    # Constants
    "UNKNOWN_FALLBACK_REPLY",
    "LOW_CONFIDENCE_FALLBACK",
    "SMALL_TALK_FALLBACK_REPLY",
    "DEFAULT_QUICK_REPLY_PRESETS",
    "SYMPTOM_PARAM_ALIASES",
    "SYMPTOM_FIELD_QUESTIONS",
    "SYMPTOM_EXTRA_HINT",
    "PRODUCT_RESULT_LIMIT",
    # Functions
    "default_quick_replies",
    "symptom_clarification_quick_replies",
    "no_results_quick_replies",
    "merge_quick_replies",
    "has_non_empty_value",
    "missing_symptom_parameters",
    "build_symptom_clarification_text",
    "no_results_reply_text",
    "deduplicate_products",
    "build_purchase_history_quick_replies",
    "most_frequent_item",
]

