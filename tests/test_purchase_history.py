"""Tests for Purchase History Context feature."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.models.purchase_history import PurchaseFrequency
from app.services.purchase_history_service import PurchaseHistoryService


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()

    def _get_settings_override():
        return Settings(openai_api_key="", use_langchain=False)

    app.dependency_overrides[get_settings] = _get_settings_override
    return TestClient(app)


@pytest.fixture
def history_service():
    """Create purchase history service."""
    settings = Settings(openai_api_key="", use_langchain=False)
    return PurchaseHistoryService(settings)


def test_get_personalization_context_returning_customer(client: TestClient) -> None:
    """Test personalization for returning customer."""
    resp = client.post(
        "/api/product-ai/personalization/context",
        json={
            "user_id": "user-123",
            "product_id": "vit-d3-001",
            "product_name": "Витамин D3 Аквадетрим",
        },
    )
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["user_id"] == "user-123"
    assert data["product_id"] == "vit-d3-001"
    
    # User has purchased this product before
    profile = data["profile"]
    assert profile["current_product_purchase_count"] > 0
    assert profile["is_returning_customer"] is True


def test_get_personalization_context_new_user(client: TestClient) -> None:
    """Test personalization for new user."""
    resp = client.post(
        "/api/product-ai/personalization/context",
        json={
            "user_id": "new-user-999",
            "product_id": "some-product",
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    profile = data["profile"]
    assert profile["current_product_purchase_count"] == 0
    assert profile["current_product_frequency"] == "first_time"
    assert profile["is_returning_customer"] is False


def test_get_personalization_context_frequent_buyer(client: TestClient) -> None:
    """Test personalization for frequent buyer."""
    resp = client.post(
        "/api/product-ai/personalization/context",
        json={
            "user_id": "user-456",
            "product_id": "ator-002",  # This user buys this frequently
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    profile = data["profile"]
    assert profile["current_product_purchase_count"] >= 3
    assert profile["current_product_frequency"] in ["regular", "frequent"]


def test_personalization_includes_message(client: TestClient) -> None:
    """Test that personalization includes a message for returning customers."""
    resp = client.post(
        "/api/product-ai/personalization/context",
        json={
            "user_id": "user-123",
            "product_id": "vit-d3-001",
            "product_name": "Витамин D3",
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert data["personalized_message"] is not None
    assert len(data["personalized_message"]) > 0


def test_personalization_suggests_quantity_for_frequent(client: TestClient) -> None:
    """Test that suggested quantity is provided for frequent buyers."""
    resp = client.post(
        "/api/product-ai/personalization/context",
        json={
            "user_id": "user-456",
            "product_id": "ator-002",
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    # Frequent buyers should get quantity suggestion
    if data["profile"]["current_product_frequency"] == "frequent":
        assert data["suggested_quantity"] is not None


def test_personalization_also_bought(client: TestClient) -> None:
    """Test that also_bought products are returned."""
    resp = client.post(
        "/api/product-ai/personalization/context",
        json={
            "user_id": "user-123",
            "product_id": "vit-d3-001",
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    # User has bought other products
    assert len(data["also_bought"]) > 0


def test_service_determine_frequency(history_service: PurchaseHistoryService) -> None:
    """Test frequency determination."""
    assert history_service._determine_frequency(0) == PurchaseFrequency.FIRST_TIME
    assert history_service._determine_frequency(1) == PurchaseFrequency.OCCASIONAL
    assert history_service._determine_frequency(2) == PurchaseFrequency.OCCASIONAL
    assert history_service._determine_frequency(3) == PurchaseFrequency.REGULAR
    assert history_service._determine_frequency(5) == PurchaseFrequency.REGULAR
    assert history_service._determine_frequency(6) == PurchaseFrequency.FREQUENT


@pytest.mark.asyncio
async def test_service_get_personalization_context(history_service: PurchaseHistoryService) -> None:
    """Test the personalization context method."""
    context = await history_service.get_personalization_context(
        user_id="user-123",
        product_id="vit-d3-001",
    )

    assert context.user_id == "user-123"
    assert context.product_id == "vit-d3-001"
    assert context.has_purchased_before is True
    assert context.purchase_count > 0
    assert context.greeting_type in ["new", "returning", "regular"]


@pytest.mark.asyncio
async def test_service_get_personalization_context_new_user(history_service: PurchaseHistoryService) -> None:
    """Test personalization context for new user."""
    context = await history_service.get_personalization_context(
        user_id="unknown-user",
        product_id="any-product",
    )

    assert context.has_purchased_before is False
    assert context.purchase_count == 0
    assert context.frequency == PurchaseFrequency.FIRST_TIME
    assert context.greeting_type == "new"
    assert context.can_reference_history is False
