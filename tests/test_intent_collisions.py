from __future__ import annotations

import pytest

from app.intents import IntentType
from app.models import ChatRequest
from app.services.router import RouterService


@pytest.fixture(scope="module")
def router() -> RouterService:
    return RouterService()


@pytest.mark.parametrize(
    "message,expected_intent",
    [
        ("покажи корзину", IntentType.SHOW_CART),
        ("добавь нурофен в корзину", IntentType.ADD_TO_CART),
        ("убери нурофен из корзины", IntentType.REMOVE_FROM_CART),
        ("очисти корзину", IntentType.CLEAR_CART),
        ("покажи избранное", IntentType.SHOW_FAVORITES),
        ("добавь нурофен в избранное", IntentType.ADD_TO_FAVORITES),
        ("мой профиль", IntentType.SHOW_PROFILE),
        ("обнови профиль, поменяй телефон", IntentType.UPDATE_PROFILE),
        ("применить промокод SAVE10", IntentType.APPLY_PROMO_CODE),
        ("мои промокоды", IntentType.SHOW_ACTIVE_COUPONS),
        ("аналоги нурофена", IntentType.FIND_ANALOGS),
        ("инструкция нурофен", IntentType.SHOW_PRODUCT_INSTRUCTIONS),
        ("где купить нурофен", IntentType.SHOW_PRODUCT_AVAILABILITY),
    ],
)
def test_router_resolves_collisions(router: RouterService, message: str, expected_intent: IntentType) -> None:
    result = router.match(
        request=ChatRequest(message=message, conversation_id="c-test"),
        user_profile=None,
        dialog_state=None,
        debug_builder=None,
    )

    assert result.matched is True, f"Router did not match for '{message}'"
    assert result.intent == expected_intent

    lower_msg = message.lower()
    if "нурофен" in lower_msg:
        assert "product_name" in result.slots
        assert "нурофен" in result.slots.get("product_name", "").lower()

    if expected_intent == IntentType.APPLY_PROMO_CODE:
        assert result.slots.get("promo_code") == "SAVE10"


def test_router_extracts_quantity(router: RouterService) -> None:
    result = router.match(
        request=ChatRequest(message="добавь 2 нурофена в корзину", conversation_id="c-qty"),
        user_profile=None,
        dialog_state=None,
        debug_builder=None,
    )

    assert result.intent == IntentType.ADD_TO_CART
    assert result.slots.get("qty") == 2
    assert result.slots.get("product_name")


def test_router_category_and_inn(router: RouterService) -> None:
    category_result = router.match(
        request=ChatRequest(message="витамины для детей", conversation_id="c-cat"),
        user_profile=None,
        dialog_state=None,
        debug_builder=None,
    )
    assert category_result.intent == IntentType.FIND_BY_CATEGORY
    assert category_result.slots.get("category")

    inn_result = router.match(
        request=ChatRequest(message="подбери по инн ибупрофен", conversation_id="c-inn"),
        user_profile=None,
        dialog_state=None,
        debug_builder=None,
    )
    assert inn_result.intent == IntentType.FIND_PRODUCT_BY_INN
    assert inn_result.slots.get("inn")

