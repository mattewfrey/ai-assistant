from __future__ import annotations

from app.intents import IntentType
from app.models import ChatRequest
from app.services.local_router import route


def test_local_router_cart():
    result = route(ChatRequest(message="Покажи корзину"))
    assert result.matched is True
    assert result.intent == IntentType.SHOW_CART


def test_local_router_orders_history():
    result = route(ChatRequest(message="История заказов"))
    assert result.matched is True
    assert result.intent == IntentType.SHOW_ORDER_HISTORY


def test_local_router_favorites_and_profile():
    fav = route(ChatRequest(message="Избранное"))
    profile = route(ChatRequest(message="Мой профиль"))

    assert fav.intent == IntentType.SHOW_FAVORITES
    assert profile.intent == IntentType.SHOW_PROFILE


def test_local_router_product_name_short_query():
    result = route(ChatRequest(message="Нурофен экспресс"))

    assert result.matched is True
    assert result.intent == IntentType.FIND_PRODUCT_BY_NAME
    assert result.parameters["product_name"] == "Нурофен экспресс"
    assert result.parameters["name"] == "Нурофен экспресс"


def test_local_router_ignores_long_freeform_phrase():
    result = route(ChatRequest(message="Посоветуй что-нибудь от головной боли до 500 рублей"))

    assert result.matched is False


def test_local_router_does_not_intercept_actions():
    result = route(ChatRequest(message="добавь нурофен в корзину"))
    assert result.matched is False
