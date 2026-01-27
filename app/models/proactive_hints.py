"""Models for proactive chat hints and triggers."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProactiveTriggerType(StrEnum):
    """Types of proactive triggers that can activate hints."""
    
    TIME_ON_PAGE = "time_on_page"  # User has been on page for X seconds
    SCROLL_DEPTH = "scroll_depth"  # User scrolled to X% of page
    EXIT_INTENT = "exit_intent"    # User is about to leave
    IDLE = "idle"                  # User has been idle for X seconds
    RETURN_VISIT = "return_visit"  # User returned to the same product
    CART_HESITATION = "cart_hesitation"  # User has product in cart but not checking out


class ProactiveHintType(StrEnum):
    """Types of proactive hints to show to user."""
    
    FAQ_SUGGESTION = "faq_suggestion"        # Suggest a common question
    AVAILABILITY_ALERT = "availability_alert"  # Alert about stock/availability
    PRICE_INFO = "price_info"                # Proactive price/discount info
    HELP_OFFER = "help_offer"                # Generic help offer
    DELIVERY_INFO = "delivery_info"          # Delivery time/options info
    COMPARISON_HINT = "comparison_hint"      # Hint about comparing products


class ProactiveTriggerContext(BaseModel):
    """Context about user behavior that triggers proactive hints."""
    
    model_config = ConfigDict(extra="ignore")

    product_id: str
    trigger_type: ProactiveTriggerType
    time_on_page_seconds: Optional[int] = None
    scroll_depth_percent: Optional[int] = None
    is_return_visit: bool = False
    has_in_cart: bool = False
    store_id: Optional[str] = None
    shipping_method: Optional[str] = None
    user_id: Optional[str] = None


class ProactiveHint(BaseModel):
    """A single proactive hint to show to user."""
    
    model_config = ConfigDict(extra="forbid")

    hint_type: ProactiveHintType
    message: str = Field(..., description="The hint message to display")
    suggested_question: Optional[str] = Field(None, description="Pre-filled question for chat")
    priority: int = Field(default=5, ge=1, le=10, description="Display priority (10 = highest)")
    dismissible: bool = Field(default=True, description="Whether user can dismiss this hint")
    action_label: Optional[str] = Field(None, description="Label for action button if any")
    metadata: Optional[Dict[str, Any]] = None


class ProactiveHintsRequest(BaseModel):
    """Request for proactive hints."""
    
    model_config = ConfigDict(extra="ignore")

    product_id: str
    trigger_type: ProactiveTriggerType
    context: Optional[ProactiveTriggerContext] = None
    user_id: Optional[str] = None
    store_id: Optional[str] = None
    limit: int = Field(default=3, ge=1, le=10)


class ProactiveHintsResponse(BaseModel):
    """Response with proactive hints for a product."""
    
    product_id: str
    hints: List[ProactiveHint] = Field(default_factory=list)
    trigger_type: ProactiveTriggerType
    meta: Optional[Dict[str, Any]] = None
