"""Models for Smart Analogs feature - finding cheaper alternatives by INN."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AnalogType(StrEnum):
    """Type of analog relationship."""
    
    SAME_INN = "same_inn"           # Same active ingredient (generic)
    SIMILAR_ACTION = "similar_action"  # Similar therapeutic action
    SAME_CATEGORY = "same_category"    # Same product category


class ProductAnalog(BaseModel):
    """A product analog with comparison info."""
    
    model_config = ConfigDict(extra="ignore")

    product_id: str
    name: str
    manufacturer: Optional[str] = None
    
    # Pricing
    price: Optional[float] = None
    price_no_discount: Optional[float] = None
    
    # Comparison to original
    price_difference: Optional[float] = Field(None, description="Difference from original price")
    savings_percent: Optional[float] = Field(None, description="Percent savings compared to original")
    
    # Analog details
    analog_type: AnalogType = AnalogType.SAME_INN
    active_ingredient: Optional[str] = Field(None, description="INN/МНН")
    dosage: Optional[str] = None
    form: Optional[str] = None
    
    # Availability
    in_stock: bool = True
    pickup_stores_count: Optional[int] = None
    
    # Quality indicators
    prescription_required: bool = False


class SmartAnalogsRequest(BaseModel):
    """Request for finding smart analogs."""
    
    model_config = ConfigDict(extra="ignore")

    product_id: str
    product_name: Optional[str] = None
    active_ingredient: Optional[str] = Field(None, description="INN/МНН to search for")
    
    # Filters
    max_price: Optional[float] = None
    in_stock_only: bool = True
    same_form_only: bool = False
    
    # Context
    store_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Limits
    limit: int = Field(default=5, ge=1, le=20)


class SmartAnalogsResponse(BaseModel):
    """Response with smart analogs for a product."""
    
    product_id: str
    product_name: Optional[str] = None
    product_price: Optional[float] = None
    active_ingredient: Optional[str] = None
    
    analogs: List[ProductAnalog] = Field(default_factory=list)
    
    # Summary
    cheapest_price: Optional[float] = None
    max_savings_percent: Optional[float] = None
    analogs_count: int = 0
    
    meta: Optional[Dict[str, Any]] = None


class AnalogComparisonItem(BaseModel):
    """Item for side-by-side comparison."""
    
    attribute: str
    original_value: Optional[str] = None
    analog_value: Optional[str] = None
    is_same: bool = False
