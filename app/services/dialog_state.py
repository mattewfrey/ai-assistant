from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..intents import ActionChannel, IntentType


class DialogState(BaseModel):
    """Lightweight dialog snapshot used for slot-filling without LLM."""

    current_intent: IntentType | None = None
    channel: ActionChannel | None = None
    slots: Dict[str, Any] = Field(default_factory=dict)
    pending_slots: List[str] = Field(default_factory=list)
    slot_questions: Dict[str, str] = Field(default_factory=dict)
    last_prompt: Optional[str] = None
    context_products: List[Dict[str, Any]] = Field(default_factory=list)
    last_reply: Optional[str] = None

    def merged_slots(self, **new_values: Any) -> Dict[str, Any]:
        merged = {**self.slots}
        merged.update({k: v for k, v in new_values.items() if v is not None})
        return merged
