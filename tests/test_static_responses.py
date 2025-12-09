"""Tests for static responses module."""

from __future__ import annotations

import pytest

from app.intents import IntentType
from app.services.static_responses import (
    LEGAL_RESPONSES,
    get_legal_response,
    is_legal_intent,
)


class TestGetLegalResponse:
    """Tests for get_legal_response function."""

    def test_return_policy_response(self) -> None:
        """Test RETURN_POLICY returns proper content."""
        response = get_legal_response(IntentType.RETURN_POLICY)

        assert response is not None
        assert "возврат" in response.lower() or "обмен" in response.lower()
        assert len(response) > 100  # Should be substantial text

    def test_storage_rules_response(self) -> None:
        """Test STORAGE_RULES returns proper content."""
        response = get_legal_response(IntentType.STORAGE_RULES)

        assert response is not None
        assert "хранени" in response.lower() or "температур" in response.lower()

    def test_expiration_info_response(self) -> None:
        """Test EXPIRATION_INFO returns proper content."""
        response = get_legal_response(IntentType.EXPIRATION_INFO)

        assert response is not None
        assert "срок" in response.lower() or "годност" in response.lower()

    def test_prescription_policy_response(self) -> None:
        """Test PRESCRIPTION_POLICY returns proper content."""
        response = get_legal_response(IntentType.PRESCRIPTION_POLICY)

        assert response is not None
        assert "рецепт" in response.lower()

    def test_delivery_rules_response(self) -> None:
        """Test DELIVERY_RULES returns proper content."""
        response = get_legal_response(IntentType.DELIVERY_RULES)

        assert response is not None
        assert "доставк" in response.lower() or "самовывоз" in response.lower()

    def test_pharmacy_legal_info_response(self) -> None:
        """Test PHARMACY_LEGAL_INFO returns proper content."""
        response = get_legal_response(IntentType.PHARMACY_LEGAL_INFO)

        assert response is not None
        assert "лицензи" in response.lower() or "юридическ" in response.lower()

    def test_safety_warnings_response(self) -> None:
        """Test SAFETY_WARNINGS returns proper content."""
        response = get_legal_response(IntentType.SAFETY_WARNINGS)

        assert response is not None
        assert "безопасност" in response.lower() or "предупрежд" in response.lower()

    def test_selfcare_advice_response(self) -> None:
        """Test SYMPTOM_SELFCARE_ADVICE returns proper content."""
        response = get_legal_response(IntentType.SYMPTOM_SELFCARE_ADVICE)

        assert response is not None
        assert "рекомендац" in response.lower() or "врач" in response.lower()

    def test_prevention_advice_response(self) -> None:
        """Test PREVENTION_ADVICE returns proper content."""
        response = get_legal_response(IntentType.PREVENTION_ADVICE)

        assert response is not None
        assert "профилактик" in response.lower() or "иммунитет" in response.lower()

    def test_ask_pharmacist_response(self) -> None:
        """Test ASK_PHARMACIST returns proper content."""
        response = get_legal_response(IntentType.ASK_PHARMACIST)

        assert response is not None
        assert "фармацевт" in response.lower() or "консультац" in response.lower()

    def test_unknown_intent_returns_none(self) -> None:
        """Test that non-legal intents return None."""
        response = get_legal_response(IntentType.SHOW_CART)

        assert response is None

    def test_find_by_symptom_returns_none(self) -> None:
        """Test that product intents return None."""
        response = get_legal_response(IntentType.FIND_BY_SYMPTOM)

        assert response is None


class TestIsLegalIntent:
    """Tests for is_legal_intent function."""

    @pytest.mark.parametrize(
        "intent",
        [
            IntentType.RETURN_POLICY,
            IntentType.STORAGE_RULES,
            IntentType.EXPIRATION_INFO,
            IntentType.PRESCRIPTION_POLICY,
            IntentType.DELIVERY_RULES,
            IntentType.PHARMACY_LEGAL_INFO,
            IntentType.SAFETY_WARNINGS,
            IntentType.SYMPTOM_SELFCARE_ADVICE,
            IntentType.PREVENTION_ADVICE,
            IntentType.ASK_PHARMACIST,
        ],
    )
    def test_legal_intents_return_true(self, intent: IntentType) -> None:
        """Test that legal intents are recognized."""
        assert is_legal_intent(intent) is True

    @pytest.mark.parametrize(
        "intent",
        [
            IntentType.SHOW_CART,
            IntentType.ADD_TO_CART,
            IntentType.FIND_BY_SYMPTOM,
            IntentType.SHOW_ORDER_STATUS,
            IntentType.PLACE_ORDER,
            IntentType.UNKNOWN,
            IntentType.SMALL_TALK,
        ],
    )
    def test_non_legal_intents_return_false(self, intent: IntentType) -> None:
        """Test that non-legal intents are not recognized as legal."""
        assert is_legal_intent(intent) is False


class TestLegalResponsesCompleteness:
    """Tests for completeness of legal responses."""

    def test_all_responses_have_content(self) -> None:
        """Test that all defined responses have meaningful content."""
        for intent, response in LEGAL_RESPONSES.items():
            assert response, f"Response for {intent} should not be empty"
            assert len(response) > 50, f"Response for {intent} should be substantial"

    def test_all_responses_are_stripped(self) -> None:
        """Test that responses don't have leading/trailing whitespace."""
        for intent, response in LEGAL_RESPONSES.items():
            assert response == response.strip(), f"Response for {intent} should be stripped"

    def test_responses_contain_medical_disclaimer_hints(self) -> None:
        """Test that medical-related responses mention professional consultation."""
        medical_intents = [
            IntentType.SYMPTOM_SELFCARE_ADVICE,
            IntentType.PREVENTION_ADVICE,
            IntentType.ASK_PHARMACIST,
            IntentType.SAFETY_WARNINGS,
        ]

        for intent in medical_intents:
            response = LEGAL_RESPONSES.get(intent, "")
            # Should mention doctor or pharmacist consultation
            assert (
                "врач" in response.lower()
                or "специалист" in response.lower()
                or "фармацевт" in response.lower()
            ), f"Response for {intent} should mention professional consultation"

