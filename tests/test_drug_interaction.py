"""Tests for Drug Interaction Checker feature."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.models.drug_interaction import DrugInfo, InteractionSeverity
from app.services.drug_interaction_service import DrugInteractionService


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()

    def _get_settings_override():
        return Settings(openai_api_key="", use_langchain=False)

    app.dependency_overrides[get_settings] = _get_settings_override
    return TestClient(app)


@pytest.fixture
def drug_service():
    """Create drug interaction service."""
    settings = Settings(openai_api_key="", use_langchain=False)
    return DrugInteractionService(settings)


def test_check_interactions_major(client: TestClient) -> None:
    """Test detection of major drug interaction."""
    resp = client.post(
        "/api/product-ai/drug-interactions/check",
        json={
            "product_id": "prod-123",
            "product_name": "Нурофен 400мг",
            "active_ingredient": "ибупрофен",
            "other_drugs": [
                {"name": "Кардиомагнил", "active_ingredient": "ацетилсалициловая кислота"}
            ],
        },
    )
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["product_id"] == "prod-123"
    assert len(data["interactions"]) > 0
    assert data["has_major_interaction"] is True

    # Check interaction details
    interaction = data["interactions"][0]
    assert interaction["severity"] in ["major", "contraindicated"]
    assert "рекомендация" in interaction["recommendation"].lower() or "избегайте" in interaction["recommendation"].lower()


def test_check_interactions_no_interaction(client: TestClient) -> None:
    """Test when no interaction is found."""
    resp = client.post(
        "/api/product-ai/drug-interactions/check",
        json={
            "product_id": "prod-456",
            "product_name": "Витамин D3",
            "active_ingredient": "колекальциферол",
            "other_drugs": [
                {"name": "Омепразол", "active_ingredient": "омепразол"}
            ],
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["interactions"]) == 0
    assert data["has_major_interaction"] is False


def test_check_interactions_contraindicated(client: TestClient) -> None:
    """Test detection of contraindicated interaction."""
    resp = client.post(
        "/api/product-ai/drug-interactions/check",
        json={
            "product_id": "prod-789",
            "product_name": "Трихопол",
            "active_ingredient": "метронидазол",
            "other_drugs": [
                {"name": "Алкоголь", "active_ingredient": "алкоголь"}
            ],
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["interactions"]) > 0
    assert data["has_contraindication"] is True

    interaction = data["interactions"][0]
    assert interaction["severity"] == "contraindicated"


def test_check_interactions_trade_name_resolution(client: TestClient) -> None:
    """Test that trade names are resolved to active ingredients."""
    resp = client.post(
        "/api/product-ai/drug-interactions/check",
        json={
            "product_id": "prod-101",
            "product_name": "Нурофен",  # Trade name for ibuprofen
            "other_drugs": [
                {"name": "Кардиомагнил"}  # Trade name for aspirin
            ],
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    # Should find interaction between ibuprofen and aspirin
    assert len(data["interactions"]) > 0


def test_check_multiple_interactions(client: TestClient) -> None:
    """Test checking multiple drugs at once."""
    resp = client.post(
        "/api/product-ai/drug-interactions/check",
        json={
            "product_id": "prod-multi",
            "product_name": "Варфарин",
            "active_ingredient": "варфарин",
            "other_drugs": [
                {"name": "Аспирин", "active_ingredient": "ацетилсалициловая кислота"},
                {"name": "Ибупрофен", "active_ingredient": "ибупрофен"},
                {"name": "Парацетамол", "active_ingredient": "парацетамол"},
            ],
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert data["checked_drugs_count"] == 3
    # Warfarin interacts with both aspirin and ibuprofen
    assert len(data["interactions"]) >= 2


def test_drug_service_normalize_name(drug_service: DrugInteractionService) -> None:
    """Test drug name normalization."""
    # Trade name to ingredient
    assert drug_service._normalize_drug_name("Нурофен") == "ибупрофен"
    assert drug_service._normalize_drug_name("Панадол") == "парацетамол"
    assert drug_service._normalize_drug_name("Кардиомагнил") == "ацетилсалициловая кислота"

    # Strip dosage info
    assert drug_service._normalize_drug_name("Нурофен 400 мг") == "ибупрофен"

    # Unknown name passes through
    assert drug_service._normalize_drug_name("НеизвестныйПрепарат") == "неизвестныйпрепарат"


def test_drug_service_get_common_interactions(drug_service: DrugInteractionService) -> None:
    """Test getting common interactions for a drug."""
    interactions = drug_service.get_common_interactions_for_drug("варфарин")

    assert len(interactions) > 0
    # Warfarin has interactions with aspirin and ibuprofen
    other_drugs = [i.drug_b.lower() for i in interactions]
    assert any("аспирин" in d or "ибупрофен" in d for d in other_drugs)


@pytest.mark.asyncio
async def test_drug_service_check_single_interaction(drug_service: DrugInteractionService) -> None:
    """Test checking a single interaction."""
    interaction = await drug_service.check_single_interaction(
        drug_a="метформин",
        drug_b="алкоголь",
    )

    assert interaction is not None
    assert interaction.severity == InteractionSeverity.MAJOR
    assert "лактоацидоз" in interaction.description.lower()


@pytest.mark.asyncio
async def test_drug_service_no_interaction(drug_service: DrugInteractionService) -> None:
    """Test when no interaction exists."""
    interaction = await drug_service.check_single_interaction(
        drug_a="витамин с",
        drug_b="магний",
    )

    assert interaction is None
