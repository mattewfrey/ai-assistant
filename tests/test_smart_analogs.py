"""Tests for Smart Analogs feature."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.services.smart_analogs_service import SmartAnalogsService


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()

    def _get_settings_override():
        return Settings(openai_api_key="", use_langchain=False)

    app.dependency_overrides[get_settings] = _get_settings_override
    return TestClient(app)


@pytest.fixture
def analogs_service():
    """Create smart analogs service."""
    settings = Settings(openai_api_key="", use_langchain=False)
    return SmartAnalogsService(settings)


def test_find_analogs_by_inn(client: TestClient) -> None:
    """Test finding analogs by active ingredient."""
    resp = client.post(
        "/api/product-ai/analogs/find",
        json={
            "product_id": "ibu-003",
            "product_name": "Нурофен Экспресс",
            "active_ingredient": "ибупрофен",
            "limit": 5,
        },
    )
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["product_id"] == "ibu-003"
    assert data["active_ingredient"] == "ибупрофен"
    assert len(data["analogs"]) > 0

    # All analogs should have the same INN
    for analog in data["analogs"]:
        assert analog["active_ingredient"] == "ибупрофен"


def test_find_analogs_sorted_by_price(client: TestClient) -> None:
    """Test that analogs are sorted by price (cheapest first)."""
    resp = client.post(
        "/api/product-ai/analogs/find",
        json={
            "product_id": "ibu-003",
            "active_ingredient": "ибупрофен",
            "limit": 10,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    prices = [a["price"] for a in data["analogs"] if a["price"]]
    assert prices == sorted(prices)  # Should be ascending


def test_find_analogs_calculates_savings(client: TestClient) -> None:
    """Test that savings are calculated correctly."""
    resp = client.post(
        "/api/product-ai/analogs/find",
        json={
            "product_id": "ibu-003",
            "product_name": "Нурофен Экспресс",
            "active_ingredient": "ибупрофен",
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    # Should have max_savings_percent if there are cheaper alternatives
    if data["cheapest_price"] and data["product_price"]:
        if data["cheapest_price"] < data["product_price"]:
            assert data["max_savings_percent"] is not None
            assert data["max_savings_percent"] > 0


def test_find_analogs_by_trade_name(client: TestClient) -> None:
    """Test finding analogs when only trade name is provided."""
    resp = client.post(
        "/api/product-ai/analogs/find",
        json={
            "product_id": "test-001",
            "product_name": "Панадол 500мг",  # Trade name for paracetamol
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert data["active_ingredient"] == "парацетамол"
    assert len(data["analogs"]) > 0


def test_find_analogs_with_max_price_filter(client: TestClient) -> None:
    """Test filtering by maximum price."""
    resp = client.post(
        "/api/product-ai/analogs/find",
        json={
            "product_id": "ibu-003",
            "active_ingredient": "ибупрофен",
            "max_price": 100,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    for analog in data["analogs"]:
        assert analog["price"] <= 100


def test_find_analogs_respects_limit(client: TestClient) -> None:
    """Test that limit parameter is respected."""
    resp = client.post(
        "/api/product-ai/analogs/find",
        json={
            "product_id": "ibu-003",
            "active_ingredient": "ибупрофен",
            "limit": 2,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["analogs"]) <= 2


def test_find_analogs_no_inn_found(client: TestClient) -> None:
    """Test response when no INN can be determined."""
    resp = client.post(
        "/api/product-ai/analogs/find",
        json={
            "product_id": "unknown-001",
            "product_name": "Unknown Product",
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["analogs"]) == 0
    assert data["meta"]["reason"] == "no_inn_found"


def test_find_analogs_includes_manufacturer(client: TestClient) -> None:
    """Test that analogs include manufacturer info."""
    resp = client.post(
        "/api/product-ai/analogs/find",
        json={
            "product_id": "ibu-003",
            "active_ingredient": "ибупрофен",
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    for analog in data["analogs"]:
        assert "manufacturer" in analog
        assert analog["manufacturer"] is not None


def test_service_resolve_inn(analogs_service: SmartAnalogsService) -> None:
    """Test INN resolution from trade names."""
    assert analogs_service._resolve_inn("ибупрофен", None) == "ибупрофен"
    assert analogs_service._resolve_inn(None, "Нурофен 400мг") == "ибупрофен"
    assert analogs_service._resolve_inn(None, "Панадол") == "парацетамол"
    assert analogs_service._resolve_inn("омепразол", None) == "омепразол"
    assert analogs_service._resolve_inn(None, "Омез 20мг") == "омепразол"


@pytest.mark.asyncio
async def test_service_find_analogs(analogs_service: SmartAnalogsService) -> None:
    """Test the service directly."""
    from app.models.smart_analogs import SmartAnalogsRequest

    request = SmartAnalogsRequest(
        product_id="ator-003",
        product_name="Липримар",
        active_ingredient="аторвастатин",
    )

    response = await analogs_service.find_analogs(request=request)

    assert response.product_id == "ator-003"
    assert response.active_ingredient == "аторвастатин"
    assert len(response.analogs) > 0
    # Lipitor is expensive, should find cheaper generics
    assert response.cheapest_price is not None
    assert response.cheapest_price < 789  # Lipitor price
