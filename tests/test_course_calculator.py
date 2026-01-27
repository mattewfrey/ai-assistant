"""Tests for Course Calculator feature."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.models.course_calculator import DosageFrequency
from app.services.course_calculator_service import CourseCalculatorService


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()

    def _get_settings_override():
        return Settings(openai_api_key="", use_langchain=False)

    app.dependency_overrides[get_settings] = _get_settings_override
    return TestClient(app)


@pytest.fixture
def calculator_service():
    """Create course calculator service."""
    settings = Settings(openai_api_key="", use_langchain=False)
    return CourseCalculatorService(settings)


def test_calculate_course_basic(client: TestClient) -> None:
    """Test basic course calculation."""
    resp = client.post(
        "/api/product-ai/course/calculate",
        json={
            "product_id": "prod-123",
            "product_name": "Витамин D3 №60",
            "units_per_package": 60,
            "dose_per_intake": 1,
            "frequency": "once_daily",
            "course_days": 30,
        },
    )
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["product_id"] == "prod-123"
    assert data["total_units_needed"] == 30
    assert data["packages_needed"] == 1
    assert data["units_remaining"] == 30  # 60 - 30


def test_calculate_course_twice_daily(client: TestClient) -> None:
    """Test course calculation with twice daily dosage."""
    resp = client.post(
        "/api/product-ai/course/calculate",
        json={
            "product_id": "prod-456",
            "units_per_package": 20,
            "dose_per_intake": 1,
            "frequency": "twice_daily",
            "course_days": 7,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    # 1 * 2 * 7 = 14 units needed
    assert data["total_units_needed"] == 14
    assert data["packages_needed"] == 1
    assert data["units_remaining"] == 6  # 20 - 14


def test_calculate_course_multiple_packages(client: TestClient) -> None:
    """Test when multiple packages are needed."""
    resp = client.post(
        "/api/product-ai/course/calculate",
        json={
            "product_id": "prod-789",
            "units_per_package": 30,
            "dose_per_intake": 1,
            "frequency": "twice_daily",
            "course_days": 30,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    # 1 * 2 * 30 = 60 units needed
    assert data["total_units_needed"] == 60
    assert data["packages_needed"] == 2  # ceil(60/30) = 2
    assert data["units_remaining"] == 0


def test_calculate_course_with_reserve(client: TestClient) -> None:
    """Test course calculation with reserve percentage."""
    resp = client.post(
        "/api/product-ai/course/calculate",
        json={
            "product_id": "prod-101",
            "units_per_package": 30,
            "dose_per_intake": 1,
            "frequency": "once_daily",
            "course_days": 30,
            "add_reserve_percent": 20,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert data["packages_needed"] == 1
    assert data["packages_with_reserve"] == 2  # 30 * 1.2 = 36 units, needs 2 packages
    assert data["reserve_percent"] == 20


def test_calculate_course_every_other_day(client: TestClient) -> None:
    """Test course calculation with every-other-day frequency."""
    resp = client.post(
        "/api/product-ai/course/calculate",
        json={
            "product_id": "prod-102",
            "units_per_package": 10,
            "dose_per_intake": 1,
            "frequency": "every_other_day",
            "course_days": 30,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    # 1 * 0.5 * 30 = 15 units needed
    assert data["total_units_needed"] == 15
    assert data["packages_needed"] == 2  # ceil(15/10)


def test_calculate_course_higher_dose(client: TestClient) -> None:
    """Test course calculation with multiple units per intake."""
    resp = client.post(
        "/api/product-ai/course/calculate",
        json={
            "product_id": "prod-103",
            "units_per_package": 100,
            "dose_per_intake": 2,
            "frequency": "three_times_daily",
            "course_days": 10,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    # 2 * 3 * 10 = 60 units needed
    assert data["total_units_needed"] == 60
    assert data["packages_needed"] == 1  # 100 units per package


def test_calculate_course_includes_recommendation(client: TestClient) -> None:
    """Test that response includes a recommendation."""
    resp = client.post(
        "/api/product-ai/course/calculate",
        json={
            "product_id": "prod-104",
            "units_per_package": 30,
            "dose_per_intake": 1,
            "frequency": "once_daily",
            "course_days": 30,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert "recommendation" in data
    assert len(data["recommendation"]) > 0
    assert "30 дней" in data["recommendation"]


def test_service_extract_package_size_from_name(calculator_service: CourseCalculatorService) -> None:
    """Test package size extraction from product name."""
    context = {"product": {"name": "Витамин D3 №60 капсулы"}}
    size = calculator_service._extract_package_size(context)
    assert size == 60


def test_service_extract_package_size_from_attributes(calculator_service: CourseCalculatorService) -> None:
    """Test package size extraction from attributes."""
    context = {
        "product": {"name": "Какой-то препарат"},
        "attributes": [
            {"name": "Количество в упаковке", "value": "50 шт"},
        ],
    }
    size = calculator_service._extract_package_size(context)
    assert size == 50


@pytest.mark.asyncio
async def test_service_calculate_with_context(calculator_service: CourseCalculatorService) -> None:
    """Test calculation with product context."""
    from app.models.course_calculator import CourseCalculatorRequest

    request = CourseCalculatorRequest(
        product_id="test-001",
        course_days=30,
        frequency=DosageFrequency.ONCE_DAILY,
    )

    context = {
        "product": {"name": "Препарат №90"},
        "pricing": {"prices": [{"type": "regular", "price": 500.0}]},
    }

    result = await calculator_service.calculate(
        request=request,
        product_context=context,
    )

    assert result.units_per_package == 90  # Extracted from name
    assert result.packages_needed == 1
    assert result.price_per_package == 500.0
    assert result.total_cost == 500.0
