from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from ..intents import ActionChannel, ActionType, IntentType
from .assistant import (
    AssistantAction,
    AssistantMeta,
    AssistantResponse,
    Reply,
)


class UIState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entry_point: Optional[str] = None
    screen: Optional[str] = None
    selected_region_id: Optional[str] = None
    selected_pharmacy_id: Optional[str] = None


class ChatMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    client_version: Optional[str] = None
    device: Optional[str] = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    conversation_id: Optional[str] = None
    trace_id: Optional[str] = None
    message: str
    user_id: Optional[str] = None
    source: Optional[str] = None
    ui_state: Optional[UIState] = None
    meta: Optional[ChatMeta] = None


class UserPreferences(BaseModel):
    """User-specific filters collected from prior interactions."""

    age: Optional[int] = None
    default_max_price: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("default_max_price", "price_ceiling"),
        serialization_alias="default_max_price",
    )
    preferred_forms: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices("preferred_forms", "preferred_dosage_forms"),
        serialization_alias="preferred_forms",
    )
    sugar_free: Optional[bool] = None
    lactose_free: Optional[bool] = None
    for_children: Optional[bool] = None


class UserProfile(BaseModel):
    """Aggregated profile snapshot used for personalization."""

    user_id: str
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    tags: Optional[List[str]] = None


class DataPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    products: List[Dict[str, Any]] = Field(default_factory=list)
    cart: Optional[Dict[str, Any]] = None
    orders: List[Dict[str, Any]] = Field(default_factory=list)
    user_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices("user_profile", "profile"),
        serialization_alias="user_profile",
    )
    user_addresses: List[Dict[str, Any]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("user_addresses", "addresses"),
        serialization_alias="user_addresses",
    )
    favorites: List[Dict[str, Any]] = Field(default_factory=list)
    pharmacies: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def has_content(self) -> bool:
        return bool(
            self.products
            or self.cart
            or self.orders
            or self.user_profile
            or self.user_addresses
            or self.favorites
            or self.pharmacies
            or self.recommendations
            or self.message
            or self.metadata
        )

    def merge(self, other: "DataPayload") -> None:
        if other.products:
            self.products.extend(other.products)
        if other.cart:
            self.cart = other.cart
        if other.orders:
            self.orders.extend(other.orders)
        if other.user_profile:
            self.user_profile = other.user_profile
        if other.user_addresses:
            self.user_addresses = other.user_addresses
        if other.favorites:
            self.favorites = other.favorites
        if other.pharmacies:
            self.pharmacies = other.pharmacies
        if other.recommendations:
            self.recommendations = other.recommendations


class ChatResponse(BaseModel):
    conversation_id: str
    reply: Reply
    actions: List[AssistantAction]
    meta: Optional[AssistantMeta] = None
    data: DataPayload = Field(default_factory=DataPayload)
    ui_state: Optional[Dict[str, Any]] = None


# Import LLM intent models
from .llm_intent import (
    ExtractedSlots,
    LLMDisambiguationResult,
    LLMIntentResult,
    LLMSlotExtractionResult,
    SlotType,
    merge_router_and_llm_slots,
    slots_to_parameters,
)


# Re-export assistant models for convenience
__all__ = [
    "ActionChannel",
    "ActionType",
    "AssistantAction",
    "AssistantMeta",
    "AssistantResponse",
    "ChatMeta",
    "ChatRequest",
    "ChatResponse",
    "DataPayload",
    "ExtractedSlots",
    "IntentType",
    "LLMDisambiguationResult",
    "LLMIntentResult",
    "LLMSlotExtractionResult",
    "Reply",
    "SlotType",
    "UIState",
    "UserPreferences",
    "UserProfile",
    "merge_router_and_llm_slots",
    "slots_to_parameters",
]

