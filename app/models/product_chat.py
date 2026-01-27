from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .assistant import Reply


class ProductChatUIState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    store_id: Optional[str] = None
    shipping_method: Optional[str] = None


class ProductChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    product_id: str
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    source: Optional[str] = None
    ui_state: Optional[ProductChatUIState] = None


class ProductChatCitation(BaseModel):
    field_path: str


class ProductChatMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    confidence: Optional[float] = None
    debug: Optional[Dict[str, Any]] = None


class ProductChatResponse(BaseModel):
    conversation_id: str
    reply: Reply
    meta: Optional[ProductChatMeta] = None
    citations: Optional[List[ProductChatCitation]] = None


class ProductChatRefusalReason(StrEnum):
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    NO_DATA = "NO_DATA"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    POLICY_RESTRICTED = "POLICY_RESTRICTED"


class ProductChatLLMResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    out_of_scope: bool = False
    refusal_reason: Optional[ProductChatRefusalReason] = None
    used_fields: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

