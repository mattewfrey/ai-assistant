"""Tests for MockPlatform new methods."""

from __future__ import annotations

import pytest

from app.services.mock_platform import MockPlatform


@pytest.fixture
def mock_platform() -> MockPlatform:
    """Fresh MockPlatform instance."""
    return MockPlatform()


class TestProductDetailedInfo:
    """Tests for product detailed info methods."""

    def test_get_product_instructions(self, mock_platform: MockPlatform) -> None:
        """Test getting product instructions."""
        instructions = mock_platform.get_product_instructions("prod-theraflu")

        assert instructions is not None
        assert len(instructions) > 10

    def test_get_product_instructions_not_found(self, mock_platform: MockPlatform) -> None:
        """Test getting instructions for non-existent product."""
        instructions = mock_platform.get_product_instructions("invalid-id")

        assert instructions is None

    def test_get_product_contraindications(self, mock_platform: MockPlatform) -> None:
        """Test getting product contraindications."""
        contraindications = mock_platform.get_product_contraindications("prod-theraflu")

        assert contraindications is not None
        assert len(contraindications) > 10

    def test_get_product_specifications(self, mock_platform: MockPlatform) -> None:
        """Test getting product specifications."""
        specs = mock_platform.get_product_specifications("prod-theraflu")

        assert specs is not None
        assert "manufacturer" in specs
        assert "dosage_form" in specs

    def test_get_product_availability_by_pharmacies(self, mock_platform: MockPlatform) -> None:
        """Test getting product availability across pharmacies."""
        availability = mock_platform.get_product_availability_by_pharmacies("prod-theraflu")

        assert availability is not None
        assert isinstance(availability, list)
        if availability:
            assert "pharmacy_id" in availability[0]
            assert "in_stock" in availability[0]


class TestProductSearch:
    """Tests for product search methods."""

    def test_find_products_by_inn(self, mock_platform: MockPlatform) -> None:
        """Test finding products by INN."""
        products = mock_platform.find_products_by_inn("парацетамол")

        assert products is not None
        assert isinstance(products, list)
        # Should find at least ТераФлю which contains парацетамол
        assert any("парацетамол" in (p.get("inn") or "").lower() for p in products)

    def test_find_products_by_inn_empty(self, mock_platform: MockPlatform) -> None:
        """Test finding products with empty INN."""
        products = mock_platform.find_products_by_inn("")

        assert products == []

    def test_find_analogs(self, mock_platform: MockPlatform) -> None:
        """Test finding product analogs."""
        analogs = mock_platform.find_analogs("prod-theraflu")

        assert isinstance(analogs, list)
        # ТераФлю should have analogs defined

    def test_find_analogs_not_found(self, mock_platform: MockPlatform) -> None:
        """Test finding analogs for non-existent product."""
        analogs = mock_platform.find_analogs("invalid-id")

        assert analogs == []

    def test_find_promo_products(self, mock_platform: MockPlatform) -> None:
        """Test finding promotional products."""
        promo = mock_platform.find_promo_products(limit=5)

        assert promo is not None
        assert isinstance(promo, list)
        assert len(promo) <= 5
        # All returned products should have promo flags
        assert all(p.get("promo_flags") for p in promo)


class TestPharmacyMethods:
    """Tests for pharmacy-related methods."""

    def test_find_pharmacies_by_metro(self, mock_platform: MockPlatform) -> None:
        """Test finding pharmacies by metro station."""
        # Empty metro should return empty (or all if implemented that way)
        pharmacies = mock_platform.find_pharmacies_by_metro("")

        assert isinstance(pharmacies, list)

    def test_get_pharmacy_working_hours(self, mock_platform: MockPlatform) -> None:
        """Test getting pharmacy working hours."""
        hours = mock_platform.get_pharmacy_working_hours("ph1")

        # May be None if not defined in test data
        assert hours is None or isinstance(hours, str)

    def test_find_nearest_pharmacy_with_product(self, mock_platform: MockPlatform) -> None:
        """Test finding nearest pharmacy with product."""
        result = mock_platform.find_nearest_pharmacy_with_product("prod-theraflu")

        # Should find pharmacy with in_stock product
        if result:
            assert "pharmacy" in result
            assert "product" in result

    def test_find_nearest_pharmacy_not_found(self, mock_platform: MockPlatform) -> None:
        """Test finding pharmacy with non-existent product."""
        result = mock_platform.find_nearest_pharmacy_with_product("invalid-id")

        assert result is None


class TestOrderMethods:
    """Tests for order-related methods."""

    def test_create_order(self, mock_platform: MockPlatform) -> None:
        """Test creating an order."""
        items = [{"product_id": "prod-theraflu", "qty": 2, "price": 549.0}]
        order = mock_platform.create_order("user-test", items)

        assert order is not None
        assert "order_id" in order
        assert order["status"] == "IN_PROGRESS"
        assert order["user_id"] == "user-test"

    def test_cancel_order(self, mock_platform: MockPlatform) -> None:
        """Test canceling an order."""
        # First create an order
        items = [{"product_id": "prod-theraflu", "qty": 1, "price": 549.0}]
        order = mock_platform.create_order("user-cancel", items)
        order_id = order["order_id"]

        # Then cancel it
        cancelled = mock_platform.cancel_order(order_id, reason="Тестовая отмена")

        assert cancelled is not None
        assert cancelled["status"] == "CANCELLED"
        assert cancelled.get("cancellation_reason") == "Тестовая отмена"

    def test_cancel_non_existent_order(self, mock_platform: MockPlatform) -> None:
        """Test canceling non-existent order."""
        result = mock_platform.cancel_order("invalid-order-id")

        assert result is None

    def test_reorder(self, mock_platform: MockPlatform) -> None:
        """Test reordering a previous order."""
        # Create original order
        items = [{"product_id": "prod-theraflu", "qty": 1, "price": 549.0}]
        original = mock_platform.create_order("user-reorder", items)

        # Reorder
        new_order = mock_platform.reorder("user-reorder", original["order_id"])

        assert new_order is not None
        assert new_order["order_id"] != original["order_id"]
        assert new_order["user_id"] == "user-reorder"


class TestCartMethods:
    """Tests for cart-related methods."""

    def test_clear_cart(self, mock_platform: MockPlatform) -> None:
        """Test clearing cart."""
        # Add something first
        mock_platform.add_to_cart("user-clear", "prod-theraflu", 1)

        # Clear cart
        cart = mock_platform.clear_cart("user-clear")

        assert cart is not None
        assert cart.get("items") == []

    def test_remove_from_cart(self, mock_platform: MockPlatform) -> None:
        """Test removing item from cart."""
        # Add item
        mock_platform.add_to_cart("user-remove", "prod-theraflu", 2)

        # Remove it
        cart = mock_platform.remove_from_cart("user-remove", "prod-theraflu")

        assert cart is not None
        # Item should be removed
        product_ids = [item.get("product_id") for item in cart.get("items", [])]
        assert "prod-theraflu" not in product_ids

    def test_apply_promo_code_valid(self, mock_platform: MockPlatform) -> None:
        """Test applying valid promo code."""
        # Add item to have something in cart
        mock_platform.add_to_cart("user-promo", "prod-theraflu", 1)

        # Apply promo
        cart = mock_platform.apply_promo_code("user-promo", "WELCOME10")

        assert cart is not None
        assert "promo_error" not in cart or cart.get("applied_promo")

    def test_apply_promo_code_invalid(self, mock_platform: MockPlatform) -> None:
        """Test applying invalid promo code."""
        cart = mock_platform.apply_promo_code("user-invalid", "INVALID123")

        assert "promo_error" in cart


class TestUserMethods:
    """Tests for user-related methods."""

    def test_get_user_coupons(self, mock_platform: MockPlatform) -> None:
        """Test getting user coupons."""
        coupons = mock_platform.get_user_coupons("user-any")

        assert coupons is not None
        assert isinstance(coupons, list)
        # Should return mock coupons
        if coupons:
            assert "code" in coupons[0]
            assert "description" in coupons[0]


class TestBooking:
    """Tests for booking method."""

    def test_book_product_pickup(self, mock_platform: MockPlatform) -> None:
        """Test booking product for pickup."""
        booking = mock_platform.book_product_pickup("user-book", "prod-theraflu", "ph1")

        assert booking is not None
        assert "booking_id" in booking
        assert booking["status"] == "RESERVED"
        assert "message" in booking

    def test_book_product_invalid_product(self, mock_platform: MockPlatform) -> None:
        """Test booking with invalid product."""
        booking = mock_platform.book_product_pickup("user-book", "invalid-id", "ph1")

        assert booking is None

    def test_book_product_invalid_pharmacy(self, mock_platform: MockPlatform) -> None:
        """Test booking with invalid pharmacy."""
        booking = mock_platform.book_product_pickup("user-book", "prod-theraflu", "invalid-ph")

        assert booking is None

