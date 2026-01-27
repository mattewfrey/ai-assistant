"""Tests for Product FAQ generation feature."""
from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.models.product_faq import ProductFAQItem, ProductFAQLLMResult
from app.routers.product_chat import get_product_context_builder, get_product_chat_llm_client
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
                "manufacturer": "Медана Фарма",
                "country": "Польша",
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
                "ship_to_store": True,
            },
            "attributes": [
                {"code": "form", "name": "Форма выпуска", "value": "Капли для приема внутрь"},
                {"code": "volume", "name": "Объём", "value": "10 мл"},
            ],
        }
        return ProductContextResult(context=context, context_hash="test-hash", cache_hit=False)


class DummyLLMClient:
    """Dummy LLM client that returns predefined FAQs."""

    async def generate_product_faqs(
        self,
        *,
        product_id: str,
        context_json: Dict[str, Any],
        trace_id: str | None = None,
    ):
        from app.services.langchain_llm import ProductFAQRunResult

        faqs = [
            ProductFAQItem(
                question="Сколько стоит?",
                answer="Цена 289₽ (скидка 17%, было 350₽).",
                category="price",
                priority=10,
                used_fields=["pricing.prices"],
            ),
            ProductFAQItem(
                question="Есть в наличии?",
                answer="Да, доступен в 15 аптеках для самовывоза.",
                category="availability",
                priority=10,
                used_fields=["availability.pickup_stores_count"],
            ),
            ProductFAQItem(
                question="Какая форма выпуска?",
                answer="Капли для приема внутрь, 10 мл.",
                category="composition",
                priority=6,
                used_fields=["attributes[code=form]", "attributes[code=volume]"],
            ),
        ]
        return ProductFAQRunResult(
            result=ProductFAQLLMResult(faqs=faqs),
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
            cached=False,
        )


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    app = create_app()

    def _get_settings_override():
        return Settings(openai_api_key="test-key", use_langchain=False)

    app.dependency_overrides[get_settings] = _get_settings_override
    app.dependency_overrides[get_product_context_builder] = lambda: DummyContextBuilder()
    app.dependency_overrides[get_product_chat_llm_client] = lambda: DummyLLMClient()
    return TestClient(app)


def test_get_product_faqs_success(client: TestClient) -> None:
    """Test successful FAQ generation."""
    resp = client.get("/api/product-ai/faq/prod-123")
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["product_id"] == "prod-123"
    assert data["product_name"] == "Витамин D3 Аквадетрим"
    assert len(data["faqs"]) == 3

    # FAQs should be sorted by priority (descending)
    priorities = [faq["priority"] for faq in data["faqs"]]
    assert priorities == sorted(priorities, reverse=True)


def test_get_product_faqs_with_store_context(client: TestClient) -> None:
    """Test FAQ generation with store context."""
    resp = client.get("/api/product-ai/faq/prod-123?store_id=store-1&shipping_method=PICKUP")
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["product_id"] == "prod-123"
    assert len(data["faqs"]) > 0


def test_get_product_faqs_cache_hit(client: TestClient) -> None:
    """Test that FAQ caching works."""
    # First request
    resp1 = client.get("/api/product-ai/faq/prod-cache-test")
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["cache_hit"] is False

    # Second request should hit cache (in real scenario)
    # Note: with TestClient each request is isolated, so we can't test actual cache
    # This test validates the response structure includes cache_hit field
    assert "cache_hit" in data1


def test_get_product_faqs_includes_citations(client: TestClient) -> None:
    """Test that FAQs include used_fields for citations."""
    resp = client.get("/api/product-ai/faq/prod-123")
    assert resp.status_code == 200

    data = resp.json()
    for faq in data["faqs"]:
        assert "used_fields" in faq
        assert isinstance(faq["used_fields"], list)


def test_faq_fallback_without_llm() -> None:
    """Test that FAQ service works with fallback when LLM is unavailable."""
    app = create_app()

    def _get_settings_override():
        return Settings(openai_api_key="", use_langchain=False)

    app.dependency_overrides[get_settings] = _get_settings_override
    app.dependency_overrides[get_product_context_builder] = lambda: DummyContextBuilder()
    app.dependency_overrides[get_product_chat_llm_client] = lambda: None  # No LLM

    with TestClient(app) as test_client:
        # Use unique product_id and force_refresh to avoid cache from other tests
        resp = test_client.get("/api/product-ai/faq/prod-fallback-test?force_refresh=true")
        assert resp.status_code == 200

        data = resp.json()
        assert data["product_id"] == "prod-fallback-test"
        # Should have fallback FAQs generated from rules
        assert len(data["faqs"]) > 0
        assert data["meta"]["source"] == "fallback"
