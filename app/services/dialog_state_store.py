from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List

from ..intents import ActionChannel, IntentType
from .dialog_state import DialogState


class DialogStateStore:
    """In-memory dialog/session memory keyed by conversation_id."""

    def __init__(self) -> None:
        self._states: Dict[str, DialogState] = {}
        self._lock = Lock()

    def get_state(self, conversation_id: str) -> DialogState | None:
        if not conversation_id:
            return None
        with self._lock:
            state = self._states.get(conversation_id)
            if state is None:
                return None
            return DialogState.model_validate(state.model_dump())

    def upsert_state(
        self,
        conversation_id: str,
        *,
        current_intent: IntentType | None = None,
        channel: ActionChannel | None = None,
        slots: Dict[str, Any] | None = None,
        context_products: List[Dict[str, Any]] | None = None,
        last_reply: str | None = None,
        pending_slots: List[str] | None = None,
        slot_questions: Dict[str, str] | None = None,
        last_prompt: str | None = None,
    ) -> DialogState:
        if not conversation_id:
            raise ValueError("conversation_id is required to update dialog state")
        with self._lock:
            state = self._states.setdefault(conversation_id, DialogState())
            update_payload: Dict[str, Any] = {}
            if current_intent is not None:
                update_payload["current_intent"] = current_intent
            if channel is not None:
                update_payload["channel"] = channel
            if slots:
                merged_slots = {**state.slots, **slots}
                update_payload["slots"] = merged_slots
            if context_products is not None:
                update_payload["context_products"] = [item.copy() for item in context_products]
            if last_reply is not None:
                update_payload["last_reply"] = last_reply
            if pending_slots is not None:
                update_payload["pending_slots"] = list(pending_slots)
            if slot_questions is not None:
                update_payload["slot_questions"] = dict(slot_questions)
            if last_prompt is not None:
                update_payload["last_prompt"] = last_prompt
            state = state.model_copy(update=update_payload)
            self._states[conversation_id] = state
            return state

    def clear_state(self, conversation_id: str) -> None:
        if not conversation_id:
            return
        with self._lock:
            self._states.pop(conversation_id, None)


_dialog_state_store = DialogStateStore()


def get_dialog_state_store() -> DialogStateStore:
    return _dialog_state_store

