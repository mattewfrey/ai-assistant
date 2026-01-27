"""Models for Purchase History Context - personalization based on user history."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PurchaseFrequency(StrEnum):
    """Frequency of product purchases."""
    
    FIRST_TIME = "first_time"      # First time viewing/buying
    OCCASIONAL = "occasional"      # Bought 1-2 times
    REGULAR = "regular"            # Bought 3-5 times
    FREQUENT = "frequent"          # Bought 6+ times


class PurchaseHistoryItem(BaseModel):
    """A single purchase history item."""
    
    model_config = ConfigDict(extra="ignore")

    product_id: str
    product_name: str
    purchase_date: datetime
    quantity: int = 1
    price: Optional[float] = None
    category: Optional[str] = None
    active_ingredient: Optional[str] = None


class UserPurchaseProfile(BaseModel):
    """User's purchase profile for personalization."""
    
    model_config = ConfigDict(extra="forbid")

    user_id: str
    
    # History summary
    total_purchases: int = 0
    total_orders: int = 0
    first_purchase_date: Optional[datetime] = None
    last_purchase_date: Optional[datetime] = None
    
    # Product-specific
    purchased_product_ids: List[str] = Field(default_factory=list)
    frequent_categories: List[str] = Field(default_factory=list)
    frequent_ingredients: List[str] = Field(default_factory=list)
    
    # Current product context
    current_product_purchase_count: int = 0
    current_product_frequency: PurchaseFrequency = PurchaseFrequency.FIRST_TIME
    days_since_last_purchase: Optional[int] = None
    
    # Personalization insights
    is_returning_customer: bool = False
    loyalty_tier: Optional[str] = None


class PurchaseHistoryRequest(BaseModel):
    """Request for purchase history context."""
    
    model_config = ConfigDict(extra="ignore")

    user_id: str
    product_id: str
    product_name: Optional[str] = None
    active_ingredient: Optional[str] = None
    category: Optional[str] = None


class PurchaseHistoryResponse(BaseModel):
    """Response with purchase history context."""
    
    user_id: str
    product_id: str
    profile: UserPurchaseProfile
    
    # Personalized recommendations
    personalized_message: Optional[str] = None
    suggested_quantity: Optional[int] = None
    reorder_reminder: bool = False
    
    # Related purchases
    also_bought: List[str] = Field(default_factory=list, description="Products often bought together")
    
    meta: Optional[Dict[str, Any]] = None


class PersonalizationContext(BaseModel):
    """Context for personalizing chat responses."""
    
    user_id: str
    product_id: str
    
    # Purchase history flags
    has_purchased_before: bool = False
    purchase_count: int = 0
    frequency: PurchaseFrequency = PurchaseFrequency.FIRST_TIME
    
    # Time context
    days_since_last_purchase: Optional[int] = None
    typical_reorder_interval: Optional[int] = None
    
    # Personalization hints
    greeting_type: str = "new"  # new, returning, regular
    can_reference_history: bool = False
