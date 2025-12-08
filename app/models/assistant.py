from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..intents import ActionChannel, ActionType, IntentType


class Reply(BaseModel):
    """Textual response that will be rendered to the user."""

    model_config = ConfigDict(extra="allow")

    text: str
    tone: Optional[str] = None
    title: Optional[str] = None
    display_hints: Optional[Dict[str, Any]] = None


class AssistantAction(BaseModel):
    """Single directive produced by the assistant."""

    model_config = ConfigDict(extra="allow")

    type: ActionType
    channel: ActionChannel | None = None
    intent: Optional[IntentType] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AssistantMeta(BaseModel):
    """Auxiliary metadata returned by the assistant."""

    model_config = ConfigDict(extra="allow")

    confidence: Optional[float] = None
    legal_disclaimer: Optional[str] = None
    quick_replies: Optional[List[Dict[str, Any]]] = None
    top_intent: Optional[str] = None
    extracted_entities: Optional[Dict[str, Any]] = None
    debug: Optional[Dict[str, Any]] = None


class AssistantResponse(BaseModel):
    """Full structured response produced by the LLM layer."""

    reply: Reply
    actions: List[AssistantAction] = Field(default_factory=list)
    meta: Optional[AssistantMeta] = None

