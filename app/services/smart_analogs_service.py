"""Service for finding smart analogs (generics) by INN/МНН.

This service helps users find cheaper alternatives to medications
by matching active ingredients (INN - International Nonproprietary Name).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..models.smart_analogs import (
    AnalogType,
    ProductAnalog,
    SmartAnalogsRequest,
    SmartAnalogsResponse,
)

logger = logging.getLogger(__name__)


# Mock catalog of products by INN for demonstration
# In production, this would query the product catalog API
MOCK_PRODUCTS_BY_INN: Dict[str, List[Dict[str, Any]]] = {
    "ибупрофен": [
        {
            "product_id": "ibu-001",
            "name": "Ибупрофен-Акрихин 400мг №20",
            "manufacturer": "Акрихин",
            "price": 89.0,
            "form": "таблетки",
            "dosage": "400 мг",
            "in_stock": True,
            "pickup_stores_count": 25,
        },
        {
            "product_id": "ibu-002",
            "name": "МИГ 400 №10",
            "manufacturer": "Берлин-Хеми",
            "price": 189.0,
            "form": "таблетки",
            "dosage": "400 мг",
            "in_stock": True,
            "pickup_stores_count": 18,
        },
        {
            "product_id": "ibu-003",
            "name": "Нурофен Экспресс 200мг №20",
            "manufacturer": "Рекитт",
            "price": 329.0,
            "price_no_discount": 389.0,
            "form": "капсулы",
            "dosage": "200 мг",
            "in_stock": True,
            "pickup_stores_count": 30,
        },
        {
            "product_id": "ibu-004",
            "name": "Нурофен Форте 400мг №12",
            "manufacturer": "Рекитт",
            "price": 289.0,
            "form": "таблетки",
            "dosage": "400 мг",
            "in_stock": True,
            "pickup_stores_count": 22,
        },
    ],
    "парацетамол": [
        {
            "product_id": "para-001",
            "name": "Парацетамол 500мг №20",
            "manufacturer": "Фармстандарт",
            "price": 29.0,
            "form": "таблетки",
            "dosage": "500 мг",
            "in_stock": True,
            "pickup_stores_count": 35,
        },
        {
            "product_id": "para-002",
            "name": "Панадол 500мг №12",
            "manufacturer": "ГлаксоСмитКляйн",
            "price": 89.0,
            "form": "таблетки",
            "dosage": "500 мг",
            "in_stock": True,
            "pickup_stores_count": 28,
        },
        {
            "product_id": "para-003",
            "name": "Эффералган 500мг №16",
            "manufacturer": "UPSA",
            "price": 159.0,
            "form": "шипучие таблетки",
            "dosage": "500 мг",
            "in_stock": True,
            "pickup_stores_count": 20,
        },
    ],
    "омепразол": [
        {
            "product_id": "omep-001",
            "name": "Омепразол 20мг №30",
            "manufacturer": "Озон",
            "price": 59.0,
            "form": "капсулы",
            "dosage": "20 мг",
            "in_stock": True,
            "pickup_stores_count": 30,
        },
        {
            "product_id": "omep-002",
            "name": "Омез 20мг №30",
            "manufacturer": "Др. Редди'с",
            "price": 189.0,
            "form": "капсулы",
            "dosage": "20 мг",
            "in_stock": True,
            "pickup_stores_count": 25,
        },
        {
            "product_id": "omep-003",
            "name": "Лосек МАПС 20мг №14",
            "manufacturer": "АстраЗенека",
            "price": 489.0,
            "form": "таблетки",
            "dosage": "20 мг",
            "in_stock": True,
            "pickup_stores_count": 15,
        },
    ],
    "аторвастатин": [
        {
            "product_id": "ator-001",
            "name": "Аторвастатин 20мг №30",
            "manufacturer": "Вертекс",
            "price": 189.0,
            "form": "таблетки",
            "dosage": "20 мг",
            "in_stock": True,
            "pickup_stores_count": 20,
        },
        {
            "product_id": "ator-002",
            "name": "Торвакард 20мг №30",
            "manufacturer": "Зентива",
            "price": 389.0,
            "form": "таблетки",
            "dosage": "20 мг",
            "in_stock": True,
            "pickup_stores_count": 18,
        },
        {
            "product_id": "ator-003",
            "name": "Липримар 20мг №30",
            "manufacturer": "Пфайзер",
            "price": 789.0,
            "form": "таблетки",
            "dosage": "20 мг",
            "in_stock": True,
            "pickup_stores_count": 12,
        },
    ],
    "левотироксин": [
        {
            "product_id": "levo-001",
            "name": "L-Тироксин 100мкг №100",
            "manufacturer": "Берлин-Хеми",
            "price": 159.0,
            "form": "таблетки",
            "dosage": "100 мкг",
            "in_stock": True,
            "pickup_stores_count": 25,
        },
        {
            "product_id": "levo-002",
            "name": "Эутирокс 100мкг №100",
            "manufacturer": "Мерк",
            "price": 189.0,
            "form": "таблетки",
            "dosage": "100 мкг",
            "in_stock": True,
            "pickup_stores_count": 22,
        },
    ],
}

# Mapping of trade names to INN
TRADE_NAME_TO_INN: Dict[str, str] = {
    "нурофен": "ибупрофен",
    "миг": "ибупрофен",
    "ибуклин": "ибупрофен",
    "панадол": "парацетамол",
    "эффералган": "парацетамол",
    "тайленол": "парацетамол",
    "омез": "омепразол",
    "лосек": "омепразол",
    "торвакард": "аторвастатин",
    "липримар": "аторвастатин",
    "эутирокс": "левотироксин",
    "l-тироксин": "левотироксин",
}


class SmartAnalogsService:
    """Service for finding cheaper product analogs by INN."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._products_by_inn = MOCK_PRODUCTS_BY_INN
        self._trade_name_to_inn = TRADE_NAME_TO_INN

    async def find_analogs(
        self,
        *,
        request: SmartAnalogsRequest,
        trace_id: Optional[str] = None,
    ) -> SmartAnalogsResponse:
        """Find smart analogs for a product."""

        # Determine the INN to search for
        inn = self._resolve_inn(
            active_ingredient=request.active_ingredient,
            product_name=request.product_name,
        )

        if not inn:
            logger.info(
                "smart_analogs.no_inn product_id=%s name=%s",
                request.product_id,
                request.product_name,
            )
            return SmartAnalogsResponse(
                product_id=request.product_id,
                product_name=request.product_name,
                analogs=[],
                analogs_count=0,
                meta={"reason": "no_inn_found"},
            )

        # Get products with the same INN
        products = self._products_by_inn.get(inn.lower(), [])

        # Find the original product's price for comparison
        original_price = self._get_original_price(request.product_id, products, request.product_name)

        # Build analog list
        analogs: List[ProductAnalog] = []
        for product in products:
            # Skip the original product
            if product["product_id"] == request.product_id:
                continue

            # Apply filters
            if request.in_stock_only and not product.get("in_stock", True):
                continue
            if request.max_price and product.get("price", 0) > request.max_price:
                continue
            if request.same_form_only and request.product_name:
                # Simple form matching - in production would be more sophisticated
                pass

            # Calculate savings
            price = product.get("price")
            price_difference = None
            savings_percent = None

            if price and original_price and original_price > 0:
                price_difference = original_price - price
                if price_difference > 0:
                    savings_percent = round((price_difference / original_price) * 100, 1)

            analog = ProductAnalog(
                product_id=product["product_id"],
                name=product["name"],
                manufacturer=product.get("manufacturer"),
                price=price,
                price_no_discount=product.get("price_no_discount"),
                price_difference=price_difference,
                savings_percent=savings_percent if savings_percent and savings_percent > 0 else None,
                analog_type=AnalogType.SAME_INN,
                active_ingredient=inn,
                dosage=product.get("dosage"),
                form=product.get("form"),
                in_stock=product.get("in_stock", True),
                pickup_stores_count=product.get("pickup_stores_count"),
                prescription_required=product.get("prescription", False),
            )
            analogs.append(analog)

        # Sort by price (cheapest first)
        analogs.sort(key=lambda x: x.price or float('inf'))

        # Apply limit
        analogs = analogs[:request.limit]

        # Calculate summary
        cheapest_price = min((a.price for a in analogs if a.price), default=None)
        max_savings = max((a.savings_percent for a in analogs if a.savings_percent), default=None)

        logger.info(
            "smart_analogs.found product_id=%s inn=%s count=%d cheapest=%.2f",
            request.product_id,
            inn,
            len(analogs),
            cheapest_price or 0,
        )

        return SmartAnalogsResponse(
            product_id=request.product_id,
            product_name=request.product_name,
            product_price=original_price,
            active_ingredient=inn,
            analogs=analogs,
            cheapest_price=cheapest_price,
            max_savings_percent=max_savings,
            analogs_count=len(analogs),
            meta={
                "inn": inn,
                "source": "mock_catalog",
                "trace_id": trace_id,
            },
        )

    def _resolve_inn(
        self,
        active_ingredient: Optional[str],
        product_name: Optional[str],
    ) -> Optional[str]:
        """Resolve the INN from active ingredient or product name."""
        # First try the provided active ingredient
        if active_ingredient:
            normalized = active_ingredient.strip().lower()
            if normalized in self._products_by_inn:
                return normalized
            # Check if it's a trade name
            if normalized in self._trade_name_to_inn:
                return self._trade_name_to_inn[normalized]

        # Try to extract from product name
        if product_name:
            normalized = product_name.strip().lower()
            # Check each trade name
            for trade_name, inn in self._trade_name_to_inn.items():
                if trade_name in normalized:
                    return inn
            # Check if product name contains an INN
            for inn in self._products_by_inn.keys():
                if inn in normalized:
                    return inn

        return None

    def _get_original_price(
        self,
        product_id: str,
        products: List[Dict[str, Any]],
        product_name: Optional[str],
    ) -> Optional[float]:
        """Get the original product's price for comparison."""
        # Try to find by product_id
        for product in products:
            if product["product_id"] == product_id:
                return product.get("price")

        # Try to find by name
        if product_name:
            normalized_name = product_name.strip().lower()
            for product in products:
                if normalized_name in product["name"].lower():
                    return product.get("price")

        # Return the most expensive as default (original is usually branded)
        prices = [p.get("price") for p in products if p.get("price")]
        return max(prices) if prices else None


# Singleton instance
_smart_analogs_service: Optional[SmartAnalogsService] = None


def get_smart_analogs_service(settings: Settings) -> SmartAnalogsService:
    """Get or create smart analogs service instance."""
    global _smart_analogs_service
    if _smart_analogs_service is None:
        _smart_analogs_service = SmartAnalogsService(settings)
    return _smart_analogs_service
