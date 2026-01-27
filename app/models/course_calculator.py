"""Models for Course Calculator feature - calculate packages needed for a treatment course."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DosageFrequency(StrEnum):
    """Frequency of medication intake."""
    
    ONCE_DAILY = "once_daily"           # 1 раз в день
    TWICE_DAILY = "twice_daily"         # 2 раза в день
    THREE_TIMES_DAILY = "three_times_daily"  # 3 раза в день
    FOUR_TIMES_DAILY = "four_times_daily"   # 4 раза в день
    EVERY_OTHER_DAY = "every_other_day"     # Через день
    ONCE_WEEKLY = "once_weekly"             # 1 раз в неделю
    AS_NEEDED = "as_needed"                 # По необходимости


class CourseCalculatorRequest(BaseModel):
    """Request for course calculation."""
    
    model_config = ConfigDict(extra="ignore")

    product_id: str
    product_name: Optional[str] = None

    # Dosage info (can be extracted from product or provided by user)
    units_per_package: Optional[int] = Field(None, description="Tablets/capsules per package")
    dose_per_intake: int = Field(default=1, ge=1, description="Units per single intake")
    
    # Frequency
    frequency: DosageFrequency = DosageFrequency.ONCE_DAILY
    intakes_per_day: Optional[int] = Field(None, description="Custom intakes per day if frequency is custom")
    
    # Course duration
    course_days: int = Field(..., ge=1, le=365, description="Duration of treatment course in days")
    
    # Safety margin
    add_reserve_percent: int = Field(default=0, ge=0, le=50, description="Extra reserve percentage")


class CourseCalculatorResult(BaseModel):
    """Result of course calculation."""
    
    model_config = ConfigDict(extra="forbid")

    product_id: str
    product_name: Optional[str] = None
    
    # Input summary
    units_per_package: int
    dose_per_intake: int
    frequency: DosageFrequency
    intakes_per_day: float
    course_days: int
    
    # Calculation results
    total_units_needed: int = Field(..., description="Total units needed for the course")
    packages_needed: int = Field(..., description="Number of packages to buy")
    units_remaining: int = Field(..., description="Units remaining after course")
    
    # With reserve
    packages_with_reserve: Optional[int] = None
    reserve_percent: int = 0
    
    # Price info if available
    price_per_package: Optional[float] = None
    total_cost: Optional[float] = None
    total_cost_with_reserve: Optional[float] = None
    
    # Recommendation
    recommendation: str = Field(..., description="Human-readable recommendation")
    
    meta: Optional[Dict[str, Any]] = None


class CoursePreset(BaseModel):
    """Preset course duration for common treatments."""
    
    name: str
    description: str
    days: int
    frequency: DosageFrequency = DosageFrequency.ONCE_DAILY
    dose_per_intake: int = 1


# Common course presets for pharmacy
COMMON_COURSE_PRESETS: List[CoursePreset] = [
    CoursePreset(
        name="Курс 7 дней",
        description="Стандартный курс антибиотиков",
        days=7,
        frequency=DosageFrequency.TWICE_DAILY,
    ),
    CoursePreset(
        name="Курс 14 дней",
        description="Расширенный курс антибиотиков",
        days=14,
        frequency=DosageFrequency.TWICE_DAILY,
    ),
    CoursePreset(
        name="Курс 30 дней",
        description="Месячный курс витаминов/БАД",
        days=30,
        frequency=DosageFrequency.ONCE_DAILY,
    ),
    CoursePreset(
        name="Курс 90 дней",
        description="Трёхмесячный курс витаминов/хронических препаратов",
        days=90,
        frequency=DosageFrequency.ONCE_DAILY,
    ),
    CoursePreset(
        name="Курс 180 дней",
        description="Полугодовой курс",
        days=180,
        frequency=DosageFrequency.ONCE_DAILY,
    ),
]
