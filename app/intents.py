from __future__ import annotations

from enum import StrEnum
from typing import Literal


class IntentType(StrEnum):
    """List of intents supported by the assistant."""

    UNKNOWN = "UNKNOWN"
    SMALL_TALK = "SMALL_TALK"

    FIND_PRODUCT_BY_NAME = "FIND_PRODUCT_BY_NAME"
    FIND_PRODUCT_BY_INN = "FIND_PRODUCT_BY_INN"
    FIND_ANALOGS = "FIND_ANALOGS"
    FIND_RECOMMENDATION = "FIND_RECOMMENDATION"
    FIND_BY_CATEGORY = "FIND_BY_CATEGORY"
    FIND_BY_SYMPTOM = "FIND_BY_SYMPTOM"
    FIND_BY_DISEASE = "FIND_BY_DISEASE"
    FIND_POPULAR = "FIND_POPULAR"
    FIND_NEW = "FIND_NEW"
    FIND_PROMO = "FIND_PROMO"
    FIND_BY_META_FILTERS = "FIND_BY_META_FILTERS"
    FIND_BY_PHARMACY_AVAILABILITY = "FIND_BY_PHARMACY_AVAILABILITY"

    SHOW_PRODUCT_INFO = "SHOW_PRODUCT_INFO"
    SHOW_PRODUCT_INSTRUCTIONS = "SHOW_PRODUCT_INSTRUCTIONS"
    SHOW_PRODUCT_CONTRAINDICATIONS = "SHOW_PRODUCT_CONTRAINDICATIONS"
    COMPARE_PRODUCTS = "COMPARE_PRODUCTS"
    SHOW_PRODUCT_AVAILABILITY = "SHOW_PRODUCT_AVAILABILITY"
    SHOW_NEAREST_PHARMACY_WITH_PRODUCT = "SHOW_NEAREST_PHARMACY_WITH_PRODUCT"
    SHOW_PRODUCT_REVIEWS = "SHOW_PRODUCT_REVIEWS"
    SHOW_DETAILED_PRODUCT_SPECIFICATIONS = "SHOW_DETAILED_PRODUCT_SPECIFICATIONS"
    BOOK_PRODUCT_PICKUP = "BOOK_PRODUCT_PICKUP"

    SYMPTOM_TO_PRODUCT = "SYMPTOM_TO_PRODUCT"
    DISEASE_TO_PRODUCT = "DISEASE_TO_PRODUCT"
    SYMPTOM_SELFCARE_ADVICE = "SYMPTOM_SELFCARE_ADVICE"
    PREVENTION_ADVICE = "PREVENTION_ADVICE"
    ASK_PHARMACIST = "ASK_PHARMACIST"
    LEGAL_DISCLAIMER = "LEGAL_DISCLAIMER"

    SHOW_CART = "SHOW_CART"
    ADD_TO_CART = "ADD_TO_CART"
    REMOVE_FROM_CART = "REMOVE_FROM_CART"
    CHANGE_CART_QUANTITY = "CHANGE_CART_QUANTITY"
    CLEAR_CART = "CLEAR_CART"
    SELECT_DELIVERY_TYPE = "SELECT_DELIVERY_TYPE"
    APPLY_PROMO_CODE = "APPLY_PROMO_CODE"
    SHOW_CART_TOTAL = "SHOW_CART_TOTAL"

    PLACE_ORDER = "PLACE_ORDER"
    SHOW_ORDER_STATUS = "SHOW_ORDER_STATUS"
    SHOW_ORDER_HISTORY = "SHOW_ORDER_HISTORY"
    CANCEL_ORDER = "CANCEL_ORDER"
    EXTEND_ORDER = "EXTEND_ORDER"
    SHOW_ORDER_REASON_CANCELLATION = "SHOW_ORDER_REASON_CANCELLATION"
    SHOW_ORDER_DELIVERY_TIME = "SHOW_ORDER_DELIVERY_TIME"
    TRACK_ORDER = "TRACK_ORDER"
    REORDER_PREVIOUS = "REORDER_PREVIOUS"
    SHOW_ACTIVE_ORDERS = "SHOW_ACTIVE_ORDERS"
    SHOW_COMPLETED_ORDERS = "SHOW_COMPLETED_ORDERS"

    SHOW_PROFILE = "SHOW_PROFILE"
    UPDATE_PROFILE = "UPDATE_PROFILE"
    SHOW_BONUS_BALANCE = "SHOW_BONUS_BALANCE"
    SHOW_ACTIVE_COUPONS = "SHOW_ACTIVE_COUPONS"
    SHOW_ACTIVE_PRESCRIPTIONS = "SHOW_ACTIVE_PRESCRIPTIONS"
    SHOW_LOYALTY_STATUS = "SHOW_LOYALTY_STATUS"
    SHOW_FAVORITES = "SHOW_FAVORITES"
    ADD_TO_FAVORITES = "ADD_TO_FAVORITES"
    REMOVE_FROM_FAVORITES = "REMOVE_FROM_FAVORITES"

    SHOW_NEARBY_PHARMACIES = "SHOW_NEARBY_PHARMACIES"
    SHOW_PHARMACIES_BY_METRO = "SHOW_PHARMACIES_BY_METRO"
    SHOW_PHARMACY_INFO = "SHOW_PHARMACY_INFO"
    SHOW_PHARMACY_HOURS = "SHOW_PHARMACY_HOURS"
    SHOW_PHARMACY_AVAILABILITY = "SHOW_PHARMACY_AVAILABILITY"

    RETURN_POLICY = "RETURN_POLICY"
    STORAGE_RULES = "STORAGE_RULES"
    EXPIRATION_INFO = "EXPIRATION_INFO"
    PRESCRIPTION_POLICY = "PRESCRIPTION_POLICY"
    DELIVERY_RULES = "DELIVERY_RULES"
    PHARMACY_LEGAL_INFO = "PHARMACY_LEGAL_INFO"
    SAFETY_WARNINGS = "SAFETY_WARNINGS"
    CONTACT_SUPPORT = "CONTACT_SUPPORT"


class ActionType(StrEnum):
    """Assistant directive type."""

    CALL_PLATFORM_API = "CALL_PLATFORM_API"
    NEED_AUTH = "NEED_AUTH"
    NOOP = "NOOP"
    SHOW_UI_HINT = "SHOW_UI_HINT"


class ActionChannel(StrEnum):
    """Semantic bucket returned by the LLM for orchestration."""

    DATA = "data"
    NAVIGATION = "navigation"
    ORDER = "order"
    SUPPORT = "support"
    KNOWLEDGE = "knowledge"
    OTHER = "other"


IntentCategory = Literal["symptom", "order", "navigation", "other"]

SYMPTOM_INTENTS: set[IntentType] = {
    IntentType.SYMPTOM_TO_PRODUCT,
    IntentType.DISEASE_TO_PRODUCT,
    IntentType.FIND_BY_SYMPTOM,
    IntentType.FIND_BY_DISEASE,
    IntentType.SYMPTOM_SELFCARE_ADVICE,
    IntentType.PREVENTION_ADVICE,
}

ORDER_INTENTS: set[IntentType] = {
    IntentType.PLACE_ORDER,
    IntentType.SHOW_ORDER_STATUS,
    IntentType.SHOW_ORDER_HISTORY,
    IntentType.TRACK_ORDER,
    IntentType.SHOW_ORDER_DELIVERY_TIME,
    IntentType.CANCEL_ORDER,
    IntentType.EXTEND_ORDER,
    IntentType.SHOW_ORDER_REASON_CANCELLATION,
    IntentType.REORDER_PREVIOUS,
    IntentType.SHOW_ACTIVE_ORDERS,
    IntentType.SHOW_COMPLETED_ORDERS,
}

NAVIGATION_INTENTS: set[IntentType] = {
    IntentType.SHOW_CART,
    IntentType.SHOW_PROFILE,
    IntentType.SHOW_FAVORITES,
    IntentType.SHOW_BONUS_BALANCE,
    IntentType.SHOW_ACTIVE_COUPONS,
    IntentType.SHOW_ACTIVE_PRESCRIPTIONS,
    IntentType.SHOW_LOYALTY_STATUS,
    IntentType.SHOW_NEARBY_PHARMACIES,
    IntentType.SHOW_PHARMACIES_BY_METRO,
    IntentType.SHOW_PHARMACY_INFO,
    IntentType.SHOW_PHARMACY_HOURS,
    IntentType.SHOW_PHARMACY_AVAILABILITY,
    IntentType.SHOW_CART_TOTAL,
}

_INTENT_CATEGORY_LOOKUP: dict[IntentType, IntentCategory] = {}
for intent in SYMPTOM_INTENTS:
    _INTENT_CATEGORY_LOOKUP[intent] = "symptom"
for intent in ORDER_INTENTS:
    _INTENT_CATEGORY_LOOKUP[intent] = "order"
for intent in NAVIGATION_INTENTS:
    _INTENT_CATEGORY_LOOKUP[intent] = "navigation"


def get_intent_category(intent: str | IntentType | None) -> IntentCategory:
    """Map intent code to a coarse scenario bucket."""

    if intent is None:
        return "other"
    if isinstance(intent, IntentType):
        intent_enum = intent
    else:
        try:
            intent_enum = IntentType(intent)
        except ValueError:
            return "other"
    return _INTENT_CATEGORY_LOOKUP.get(intent_enum, "other")


def intent_descriptions() -> dict[str, str]:
    """Human readable descriptions shipped to the LLM to improve grounding."""

    return {
        IntentType.UNKNOWN.value: (
            "Use when the user request does not map to any business intent. "
            "Politely say you did not understand and offer suggestions."
        ),
        IntentType.SMALL_TALK.value: (
            "Handle greetings, jokes, or general chit-chat. "
            "Answer briefly and steer the user toward pharmacy actions with quick replies."
        ),
        IntentType.FIND_BY_SYMPTOM.value: (
            "Search products by symptom description (кашель, температура, насморк). "
            "Pair with FIND_BY_META_FILTERS when the user adds constraints."
        ),
        IntentType.FIND_BY_DISEASE.value: (
            "Find OTC recommendations for a named disease or diagnosis (например, ОРВИ, гастрит)."
        ),
        IntentType.FIND_BY_META_FILTERS.value: (
            "Apply attribute filters such as «для детей», «без сахара», «без лактозы». "
            "Use together with FIND_BY_SYMPTOM/FIND_BY_DISEASE/FIND_BY_CATEGORY to refine the results."
        ),
        IntentType.FIND_BY_CATEGORY.value: (
            "Show a curated list of products inside a concrete catalog category or подборка."
        ),
        IntentType.FIND_PRODUCT_BY_NAME.value: (
            "Search the catalog by explicit product name, бренд, или штрихкод."
        ),
        IntentType.SYMPTOM_TO_PRODUCT.value: (
            "High-level reasoning intent for symptom-based advice; still emit a CALL_PLATFORM_API action "
            "with FIND_BY_SYMPTOM when the user expects SKU suggestions."
        ),
        IntentType.DISEASE_TO_PRODUCT.value: (
            "Similar reasoning intent for diagnoses; pair it with FIND_BY_DISEASE actions targeting the catalog."
        ),
        IntentType.ADD_TO_CART.value: "Add a concrete SKU to the active cart.",
        IntentType.SHOW_ORDER_STATUS.value: "Explain the current state of an order.",
        IntentType.SHOW_ACTIVE_ORDERS.value: "Summarize orders that are still being processed or waiting for pickup.",
        IntentType.SHOW_COMPLETED_ORDERS.value: "List orders that were delivered or picked up recently.",
        IntentType.SHOW_FAVORITES.value: "Show the products saved by the user to favorites.",
        IntentType.ADD_TO_FAVORITES.value: "Add a product to the user's favorites list.",
        IntentType.REMOVE_FROM_FAVORITES.value: "Remove a product from the user's favorites list.",
        IntentType.SHOW_PROFILE.value: "Display basic profile data like name or loyalty level.",
        IntentType.SHOW_NEARBY_PHARMACIES.value: "List pharmacies close to provided geo/meta info.",
        IntentType.RETURN_POLICY.value: "Explain how the pharmacy handles returns.",
        IntentType.CONTACT_SUPPORT.value: (
            "Escalate the conversation to a live support agent when the user explicitly asks for help "
            "or when automation cannot satisfy the request."
        ),
    }

