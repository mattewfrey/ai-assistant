from __future__ import annotations

from app.intents import IntentType
from app.models import ChatRequest
from app.services.router import RouterService


def test_router_matches_cart():
    router = RouterService()
    result = router.match(
        request=ChatRequest(message="покажи корзину", conversation_id="c1"),
        user_profile=None,
        dialog_state=None,
        debug_builder=None,
    )
    assert result.matched is True
    assert result.intent == IntentType.SHOW_CART


def test_router_product_by_name():
    router = RouterService()
    result = router.match(
        request=ChatRequest(message="Нурофен 200 мг", conversation_id="c1"),
        user_profile=None,
        dialog_state=None,
        debug_builder=None,
    )
    assert result.matched is True
    assert result.intent == IntentType.FIND_PRODUCT_BY_NAME
    assert result.slots.get("name") == "Нурофен 200 мг"
from __future__ import annotations

from app.models import ChatRequest
from app.models import UserProfile
from app.services.router import RouterService


def test_router_matches_cart_intent(tmp_path):
    router = RouterService()
    request = ChatRequest(message="Покажи корзину")
    result = router.match(request=request, user_profile=None, dialog_state=None)

    assert result.matched
    assert result.intent.value == "SHOW_CART"


def test_router_detects_product_query():
    router = RouterService()
    request = ChatRequest(message="Нурофен экспресс 200")

    result = router.match(request=request, user_profile=None, dialog_state=None)

    assert result.matched
    assert result.intent.value == "FIND_PRODUCT_BY_NAME"
    assert result.slots["name"]


def test_router_prefills_slots_from_profile():
    router = RouterService()
    profile = UserProfile(user_id="u1")
    profile.preferences.age = 33
    request = ChatRequest(message="Нужны таблетки от боли")

    result = router.match(request=request, user_profile=profile, dialog_state=None)

    assert result.matched
    assert result.slots.get("age") == 33
