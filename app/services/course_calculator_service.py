"""Service for calculating medication course requirements.

Helps users determine how many packages they need for a treatment course
based on dosage, frequency, and duration.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, Optional

from ..config import Settings
from ..models.course_calculator import (
    CourseCalculatorRequest,
    CourseCalculatorResult,
    DosageFrequency,
)

logger = logging.getLogger(__name__)


# Frequency to intakes per day mapping
FREQUENCY_TO_INTAKES: Dict[DosageFrequency, float] = {
    DosageFrequency.ONCE_DAILY: 1.0,
    DosageFrequency.TWICE_DAILY: 2.0,
    DosageFrequency.THREE_TIMES_DAILY: 3.0,
    DosageFrequency.FOUR_TIMES_DAILY: 4.0,
    DosageFrequency.EVERY_OTHER_DAY: 0.5,
    DosageFrequency.ONCE_WEEKLY: 1.0 / 7.0,
    DosageFrequency.AS_NEEDED: 1.0,  # Default to once daily for calculation
}


class CourseCalculatorService:
    """Service for calculating medication course requirements."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def calculate(
        self,
        *,
        request: CourseCalculatorRequest,
        product_context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> CourseCalculatorResult:
        """Calculate course requirements based on input parameters."""

        # Extract package size from product context if not provided
        units_per_package = request.units_per_package
        price_per_package = None
        product_name = request.product_name

        if product_context:
            if not units_per_package:
                units_per_package = self._extract_package_size(product_context)
            if not product_name:
                product_name = product_context.get("product", {}).get("name")
            
            # Get price
            pricing = product_context.get("pricing", {})
            prices = pricing.get("prices", [])
            if prices:
                price_per_package = prices[0].get("price")

        # Default package size if still not found
        if not units_per_package:
            units_per_package = 30  # Common default

        # Calculate intakes per day
        intakes_per_day = request.intakes_per_day
        if intakes_per_day is None:
            intakes_per_day = FREQUENCY_TO_INTAKES.get(request.frequency, 1.0)

        # Calculate total units needed
        total_units_needed = math.ceil(
            request.dose_per_intake * intakes_per_day * request.course_days
        )

        # Calculate packages needed (round up)
        packages_needed = math.ceil(total_units_needed / units_per_package)

        # Calculate remaining units
        total_units_bought = packages_needed * units_per_package
        units_remaining = total_units_bought - total_units_needed

        # Calculate with reserve if requested
        packages_with_reserve = None
        if request.add_reserve_percent > 0:
            units_with_reserve = math.ceil(
                total_units_needed * (1 + request.add_reserve_percent / 100)
            )
            packages_with_reserve = math.ceil(units_with_reserve / units_per_package)

        # Calculate costs
        total_cost = None
        total_cost_with_reserve = None
        if price_per_package:
            total_cost = packages_needed * price_per_package
            if packages_with_reserve:
                total_cost_with_reserve = packages_with_reserve * price_per_package

        # Generate recommendation
        recommendation = self._generate_recommendation(
            packages_needed=packages_needed,
            units_remaining=units_remaining,
            units_per_package=units_per_package,
            course_days=request.course_days,
            frequency=request.frequency,
            total_cost=total_cost,
            packages_with_reserve=packages_with_reserve,
        )

        logger.info(
            "course_calculator.calculated product_id=%s days=%d packages=%d",
            request.product_id,
            request.course_days,
            packages_needed,
        )

        return CourseCalculatorResult(
            product_id=request.product_id,
            product_name=product_name,
            units_per_package=units_per_package,
            dose_per_intake=request.dose_per_intake,
            frequency=request.frequency,
            intakes_per_day=intakes_per_day,
            course_days=request.course_days,
            total_units_needed=total_units_needed,
            packages_needed=packages_needed,
            units_remaining=units_remaining,
            packages_with_reserve=packages_with_reserve,
            reserve_percent=request.add_reserve_percent,
            price_per_package=price_per_package,
            total_cost=total_cost,
            total_cost_with_reserve=total_cost_with_reserve,
            recommendation=recommendation,
            meta={
                "trace_id": trace_id,
                "package_size_source": "provided" if request.units_per_package else "extracted",
            },
        )

    def _extract_package_size(self, context: Dict[str, Any]) -> Optional[int]:
        """Extract package size from product context (attributes, name, etc.)."""
        product = context.get("product", {})
        attributes = context.get("attributes", [])

        # Try to find in attributes
        package_keywords = ["количество", "штук", "таблетки", "капсулы", "упаковка"]
        for attr in attributes:
            attr_name = (attr.get("name") or "").lower()
            attr_value = attr.get("value") or ""
            
            if any(kw in attr_name for kw in package_keywords):
                # Try to extract number
                match = re.search(r"(\d+)", str(attr_value))
                if match:
                    return int(match.group(1))

        # Try to extract from product name
        product_name = product.get("name", "")
        # Common patterns: "№30", "N30", "30 шт", "30 таб"
        patterns = [
            r"[№N](\d+)",
            r"(\d+)\s*(?:шт|таб|капс)",
            r"x\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, product_name, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def _generate_recommendation(
        self,
        packages_needed: int,
        units_remaining: int,
        units_per_package: int,
        course_days: int,
        frequency: DosageFrequency,
        total_cost: Optional[float],
        packages_with_reserve: Optional[int],
    ) -> str:
        """Generate human-readable recommendation."""
        parts = []

        # Main recommendation
        if packages_needed == 1:
            parts.append(f"Для курса {course_days} дней вам понадобится 1 упаковка")
        else:
            parts.append(f"Для курса {course_days} дней вам понадобится {packages_needed} упаковки")

        # Remaining info
        if units_remaining > 0:
            if units_remaining == 1:
                parts.append(f"(останется {units_remaining} единица)")
            elif units_remaining < 5:
                parts.append(f"(останется {units_remaining} единицы)")
            else:
                parts.append(f"(останется {units_remaining} единиц)")

        # Cost
        if total_cost:
            parts.append(f"Стоимость: {total_cost:.0f}₽")

        # Reserve recommendation
        if packages_with_reserve and packages_with_reserve > packages_needed:
            parts.append(
                f"С запасом рекомендуем {packages_with_reserve} упаковки"
            )

        return ". ".join(parts) + "."


# Singleton instance
_course_calculator_service: Optional[CourseCalculatorService] = None


def get_course_calculator_service(settings: Settings) -> CourseCalculatorService:
    """Get or create course calculator service instance."""
    global _course_calculator_service
    if _course_calculator_service is None:
        _course_calculator_service = CourseCalculatorService(settings)
    return _course_calculator_service
