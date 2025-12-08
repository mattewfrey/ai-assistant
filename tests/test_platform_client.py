from __future__ import annotations

import pytest

from app.config import Settings
from app.models import ChatRequest, UserPreferences, UserProfile
from app.services.platform_client import PlatformApiClient


@pytest.mark.asyncio
async def test_price_max_filters_products() -> None:
    client = PlatformApiClient(Settings())
    request = ChatRequest(conversation_id="conv-price", message="Нужен препарат")

    results = await client.find_by_symptom({"symptom": "кашель", "price_max": 400}, request, None)

    assert results, "Expected some products within price limit"
    assert all(product.price <= 400 for product in results if product.price is not None)


@pytest.mark.asyncio
async def test_preferred_forms_respected_from_profile() -> None:
    client = PlatformApiClient(Settings())
    request = ChatRequest(conversation_id="conv-forms", message="Нужен сироп")
    profile = UserProfile(user_id="user-forms", preferences=UserPreferences(preferred_forms=["сироп"]))

    results = await client.find_by_symptom({"symptom": "кашель"}, request, profile)

    assert results, "Expected filtered products for preferred form"
    assert all(
        "сироп" in (product.name or "").lower() or "сироп" in (product.description or "").lower()
        for product in results
    )
