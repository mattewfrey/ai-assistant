"""Service for Purchase History Context - personalization based on user purchase history.

This service provides personalization context for the product AI chat
based on the user's purchase history.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..models.purchase_history import (
    PersonalizationContext,
    PurchaseFrequency,
    PurchaseHistoryItem,
    PurchaseHistoryRequest,
    PurchaseHistoryResponse,
    UserPurchaseProfile,
)

logger = logging.getLogger(__name__)


# Mock purchase history for demonstration
# In production, this would come from an orders API
MOCK_PURCHASE_HISTORY: Dict[str, List[Dict[str, Any]]] = {
    "user-123": [
        {
            "product_id": "vit-d3-001",
            "product_name": "Витамин D3 Аквадетрим",
            "purchase_date": datetime.now() - timedelta(days=30),
            "quantity": 2,
            "price": 289.0,
            "category": "Витамины",
            "active_ingredient": "колекальциферол",
        },
        {
            "product_id": "vit-d3-001",
            "product_name": "Витамин D3 Аквадетрим",
            "purchase_date": datetime.now() - timedelta(days=90),
            "quantity": 1,
            "price": 289.0,
            "category": "Витамины",
            "active_ingredient": "колекальциферол",
        },
        {
            "product_id": "omega-3-001",
            "product_name": "Омега-3 Доппельгерц",
            "purchase_date": datetime.now() - timedelta(days=60),
            "quantity": 1,
            "price": 450.0,
            "category": "Витамины",
            "active_ingredient": "омега-3",
        },
        {
            "product_id": "para-001",
            "product_name": "Парацетамол 500мг",
            "purchase_date": datetime.now() - timedelta(days=120),
            "quantity": 2,
            "price": 29.0,
            "category": "Обезболивающие",
            "active_ingredient": "парацетамол",
        },
    ],
    "user-456": [
        {
            "product_id": "ator-002",
            "product_name": "Торвакард 20мг",
            "purchase_date": datetime.now() - timedelta(days=15),
            "quantity": 1,
            "price": 389.0,
            "category": "Сердечно-сосудистые",
            "active_ingredient": "аторвастатин",
        },
        {
            "product_id": "ator-002",
            "product_name": "Торвакард 20мг",
            "purchase_date": datetime.now() - timedelta(days=45),
            "quantity": 1,
            "price": 389.0,
            "category": "Сердечно-сосудистые",
            "active_ingredient": "аторвастатин",
        },
        {
            "product_id": "ator-002",
            "product_name": "Торвакард 20мг",
            "purchase_date": datetime.now() - timedelta(days=75),
            "quantity": 1,
            "price": 389.0,
            "category": "Сердечно-сосудистые",
            "active_ingredient": "аторвастатин",
        },
        {
            "product_id": "ator-002",
            "product_name": "Торвакард 20мг",
            "purchase_date": datetime.now() - timedelta(days=105),
            "quantity": 1,
            "price": 389.0,
            "category": "Сердечно-сосудистые",
            "active_ingredient": "аторвастатин",
        },
    ],
}


class PurchaseHistoryService:
    """Service for purchase history personalization."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._mock_history = MOCK_PURCHASE_HISTORY

    async def get_purchase_context(
        self,
        *,
        request: PurchaseHistoryRequest,
        trace_id: Optional[str] = None,
    ) -> PurchaseHistoryResponse:
        """Get purchase history context for personalization."""

        # Get user's purchase history
        history = self._get_user_history(request.user_id)

        # Build profile
        profile = self._build_profile(
            user_id=request.user_id,
            product_id=request.product_id,
            history=history,
        )

        # Generate personalized message
        personalized_message = self._generate_personalized_message(
            profile=profile,
            product_name=request.product_name,
        )

        # Calculate suggested quantity
        suggested_quantity = self._calculate_suggested_quantity(profile)

        # Check if reorder reminder is appropriate
        reorder_reminder = self._should_show_reorder_reminder(profile)

        # Get "also bought" products
        also_bought = self._get_also_bought(
            user_id=request.user_id,
            product_id=request.product_id,
            history=history,
        )

        logger.info(
            "purchase_history.context user_id=%s product_id=%s frequency=%s",
            request.user_id,
            request.product_id,
            profile.current_product_frequency.value,
        )

        return PurchaseHistoryResponse(
            user_id=request.user_id,
            product_id=request.product_id,
            profile=profile,
            personalized_message=personalized_message,
            suggested_quantity=suggested_quantity,
            reorder_reminder=reorder_reminder,
            also_bought=also_bought,
            meta={
                "trace_id": trace_id,
                "source": "mock_history",
            },
        )

    async def get_personalization_context(
        self,
        *,
        user_id: str,
        product_id: str,
    ) -> PersonalizationContext:
        """Get lightweight personalization context for chat."""
        history = self._get_user_history(user_id)
        
        # Find purchases of this product
        product_purchases = [
            h for h in history if h["product_id"] == product_id
        ]
        
        purchase_count = len(product_purchases)
        frequency = self._determine_frequency(purchase_count)
        
        # Calculate days since last purchase
        days_since_last = None
        if product_purchases:
            last_purchase = max(p["purchase_date"] for p in product_purchases)
            days_since_last = (datetime.now() - last_purchase).days
        
        # Determine greeting type
        if purchase_count == 0:
            greeting_type = "new"
        elif purchase_count <= 2:
            greeting_type = "returning"
        else:
            greeting_type = "regular"
        
        return PersonalizationContext(
            user_id=user_id,
            product_id=product_id,
            has_purchased_before=purchase_count > 0,
            purchase_count=purchase_count,
            frequency=frequency,
            days_since_last_purchase=days_since_last,
            greeting_type=greeting_type,
            can_reference_history=purchase_count > 0,
        )

    def _get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's purchase history from mock data."""
        return self._mock_history.get(user_id, [])

    def _build_profile(
        self,
        user_id: str,
        product_id: str,
        history: List[Dict[str, Any]],
    ) -> UserPurchaseProfile:
        """Build user purchase profile from history."""
        if not history:
            return UserPurchaseProfile(user_id=user_id)

        # Calculate summary stats
        total_purchases = sum(h.get("quantity", 1) for h in history)
        unique_orders = len(set(h["purchase_date"].date() for h in history))
        dates = [h["purchase_date"] for h in history]
        
        # Product-specific stats
        product_purchases = [h for h in history if h["product_id"] == product_id]
        product_count = len(product_purchases)
        frequency = self._determine_frequency(product_count)
        
        # Days since last purchase of this product
        days_since_last = None
        if product_purchases:
            last_purchase = max(p["purchase_date"] for p in product_purchases)
            days_since_last = (datetime.now() - last_purchase).days

        # Frequent categories and ingredients
        categories = [h.get("category") for h in history if h.get("category")]
        ingredients = [h.get("active_ingredient") for h in history if h.get("active_ingredient")]
        
        frequent_categories = list(set(categories))[:5]
        frequent_ingredients = list(set(ingredients))[:10]

        return UserPurchaseProfile(
            user_id=user_id,
            total_purchases=total_purchases,
            total_orders=unique_orders,
            first_purchase_date=min(dates),
            last_purchase_date=max(dates),
            purchased_product_ids=list(set(h["product_id"] for h in history)),
            frequent_categories=frequent_categories,
            frequent_ingredients=frequent_ingredients,
            current_product_purchase_count=product_count,
            current_product_frequency=frequency,
            days_since_last_purchase=days_since_last,
            is_returning_customer=len(history) > 1,
        )

    def _determine_frequency(self, purchase_count: int) -> PurchaseFrequency:
        """Determine purchase frequency based on count."""
        if purchase_count == 0:
            return PurchaseFrequency.FIRST_TIME
        elif purchase_count <= 2:
            return PurchaseFrequency.OCCASIONAL
        elif purchase_count <= 5:
            return PurchaseFrequency.REGULAR
        else:
            return PurchaseFrequency.FREQUENT

    def _generate_personalized_message(
        self,
        profile: UserPurchaseProfile,
        product_name: Optional[str],
    ) -> Optional[str]:
        """Generate personalized message based on profile."""
        name = product_name or "этот товар"
        
        if profile.current_product_frequency == PurchaseFrequency.FIRST_TIME:
            if profile.is_returning_customer:
                return f"Рады снова видеть вас! Хотите узнать больше о товаре {name}?"
            return None  # No special message for new users viewing new product
        
        elif profile.current_product_frequency == PurchaseFrequency.OCCASIONAL:
            return f"Вы уже покупали {name}. Есть вопросы по этому товару?"
        
        elif profile.current_product_frequency == PurchaseFrequency.REGULAR:
            return f"Рады, что {name} вам нравится! Могу помочь с заказом."
        
        else:  # FREQUENT
            days = profile.days_since_last_purchase or 0
            if days > 25:
                return f"Пора пополнить запас? Вы обычно покупаете {name} каждый месяц."
            return f"С возвращением! {name} — ваш постоянный выбор."

    def _calculate_suggested_quantity(self, profile: UserPurchaseProfile) -> Optional[int]:
        """Calculate suggested quantity based on purchase history."""
        if profile.current_product_frequency == PurchaseFrequency.FIRST_TIME:
            return 1
        elif profile.current_product_frequency == PurchaseFrequency.FREQUENT:
            return 2  # Suggest buying more for frequent buyers
        return None

    def _should_show_reorder_reminder(self, profile: UserPurchaseProfile) -> bool:
        """Determine if reorder reminder should be shown."""
        if profile.current_product_frequency in (
            PurchaseFrequency.REGULAR,
            PurchaseFrequency.FREQUENT,
        ):
            days = profile.days_since_last_purchase or 0
            # Show reminder if it's been 25+ days since last purchase
            return days >= 25
        return False

    def _get_also_bought(
        self,
        user_id: str,
        product_id: str,
        history: List[Dict[str, Any]],
    ) -> List[str]:
        """Get products often bought together with this product."""
        # Get other products this user has bought
        other_products = [
            h["product_name"]
            for h in history
            if h["product_id"] != product_id
        ]
        # Return unique list (max 3)
        return list(set(other_products))[:3]


# Singleton instance
_purchase_history_service: Optional[PurchaseHistoryService] = None


def get_purchase_history_service(settings: Settings) -> PurchaseHistoryService:
    """Get or create purchase history service instance."""
    global _purchase_history_service
    if _purchase_history_service is None:
        _purchase_history_service = PurchaseHistoryService(settings)
    return _purchase_history_service
