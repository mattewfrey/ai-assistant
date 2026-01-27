"""Tests for Proactive Hints feature."""
from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.models.proactive_hints import ProactiveTriggerType
from app.routers.product_chat import get_product_context_builder
from app.services.product_context_builder import ProductContextResult


class DummyContextBuilder:
    """Dummy context builder for testing."""

    async def get_context(
        self,
        *,
        product_id: str,
        store_id: str | None,
        shipping_method: str | None,
        authorization: str | None,
        trace_id: str | None,
        conversation_id: str | None,
        user_id: str | None,
    ) -> ProductContextResult:
        context: Dict[str, Any] = {
            "product": {
                "id": product_id,
                "name": "Витамин D3 Аквадетрим",
                "prescription": False,
                "delivery_available": True,
            },
            "pricing": {
                "prices": [
                    {"type": "regular", "price": 289.0, "price_no_discount": 350.0}
                ]
            },
            "availability": {
                "stocks": [{"store_id": "123", "quantity": 10}],
                "pickup_stores_count": 15,
            },
        }
        return ProductContextResult(context=context, context_hash="test-hash", cache_hit=True)


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    app = create_app()

    def _get_settings_override():
        return Settings(openai_api_key="", use_langchain=False)

    app.dependency_overrides[get_settings] = _get_settings_override
    app.dependency_overrides[get_product_context_builder] = lambda: DummyContextBuilder()
    return TestClient(app)


def test_proactive_hints_time_on_page(client: TestClient) -> None:
    """Test proactive hints for time on page trigger."""
    resp = client.post(
        "/api/product-ai/proactive/hints",
        json={
            "product_id": "prod-123",
            "trigger_type": "time_on_page",
            "limit": 3,
        },
    )
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["product_id"] == "prod-123"
    assert data["trigger_type"] == "time_on_page"
    assert len(data["hints"]) > 0
    assert len(data["hints"]) <= 3

    # Check hint structure
    hint = data["hints"][0]
    assert "hint_type" in hint
    assert "message" in hint
    assert "priority" in hint


def test_proactive_hints_exit_intent(client: TestClient) -> None:
    """Test proactive hints for exit intent trigger."""
    resp = client.post(
        "/api/product-ai/proactive/hints",
        json={
            "product_id": "prod-123",
            "trigger_type": "exit_intent",
            "limit": 5,
        },
    )
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["trigger_type"] == "exit_intent"
    assert len(data["hints"]) > 0

    # Exit intent should have high priority hints
    priorities = [h["priority"] for h in data["hints"]]
    assert max(priorities) >= 8  # Should have at least one high priority hint


def test_proactive_hints_scroll_depth(client: TestClient) -> None:
    """Test proactive hints for scroll depth trigger."""
    resp = client.post(
        "/api/product-ai/proactive/hints",
        json={
            "product_id": "prod-123",
            "trigger_type": "scroll_depth",
            "limit": 3,
        },
    )
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["trigger_type"] == "scroll_depth"
    assert len(data["hints"]) > 0


def test_proactive_hints_cart_hesitation(client: TestClient) -> None:
    """Test proactive hints for cart hesitation trigger."""
    resp = client.post(
        "/api/product-ai/proactive/hints",
        json={
            "product_id": "prod-123",
            "trigger_type": "cart_hesitation",
            "limit": 3,
        },
    )
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["trigger_type"] == "cart_hesitation"


def test_proactive_hints_with_discount_shows_price_hint(client: TestClient) -> None:
    """Test that products with discounts get price hints."""
    resp = client.post(
        "/api/product-ai/proactive/hints",
        json={
            "product_id": "prod-123",
            "trigger_type": "time_on_page",
            "limit": 5,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    hint_types = [h["hint_type"] for h in data["hints"]]
    assert "price_info" in hint_types  # Should have price hint for discounted product


def test_proactive_hints_includes_suggested_question(client: TestClient) -> None:
    """Test that hints include suggested questions for chat."""
    resp = client.post(
        "/api/product-ai/proactive/hints",
        json={
            "product_id": "prod-123",
            "trigger_type": "exit_intent",
            "limit": 5,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    # At least one hint should have a suggested question
    suggested_questions = [h.get("suggested_question") for h in data["hints"] if h.get("suggested_question")]
    assert len(suggested_questions) > 0


def test_proactive_hints_respects_limit(client: TestClient) -> None:
    """Test that limit parameter is respected."""
    resp = client.post(
        "/api/product-ai/proactive/hints",
        json={
            "product_id": "prod-123",
            "trigger_type": "exit_intent",
            "limit": 1,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["hints"]) <= 1
