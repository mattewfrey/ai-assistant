from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Dict, List, Literal, Optional

HistoryRole = Literal["user", "assistant"]


@dataclass
class HistoryMessage:
    role: HistoryRole
    content: str


@dataclass
class ConversationContext:
    messages: Deque[HistoryMessage] = field(default_factory=deque)
    summary: Optional[str] = None
    total_messages: int = 0
    last_summary_turn: int = 0


class ConversationStore:
    """Simple in-memory storage for conversation snippets and summaries."""

    def __init__(self, history_limit: int = 20) -> None:
        self._history_limit = history_limit
        self._store: Dict[str, ConversationContext] = {}
        self._lock = Lock()

    def append_message(self, conversation_id: str, role: HistoryRole, content: str) -> None:
        """Add a new message and trim history to the configured size."""

        if not conversation_id:
            return
        message = HistoryMessage(role=role, content=content)
        with self._lock:
            context = self._store.setdefault(
                conversation_id, ConversationContext(messages=deque(maxlen=self._history_limit))
            )
            if context.messages.maxlen != self._history_limit:
                context.messages = deque(context.messages, maxlen=self._history_limit)
            context.messages.append(message)
            context.total_messages += 1

    def get_history(self, conversation_id: str, limit: int = 10) -> List[HistoryMessage]:
        """Return up to `limit` most recent messages for the conversation."""

        if not conversation_id or limit <= 0:
            return []
        with self._lock:
            context = self._store.get(conversation_id)
            if not context:
                return []
            history = list(context.messages)
        return history[-limit:]

    def get_summary(self, conversation_id: str) -> Optional[str]:
        if not conversation_id:
            return None
        with self._lock:
            context = self._store.get(conversation_id)
            return context.summary if context else None

    def set_summary(self, conversation_id: str, summary: str) -> None:
        if not conversation_id:
            return
        with self._lock:
            context = self._store.setdefault(
                conversation_id, ConversationContext(messages=deque(maxlen=self._history_limit))
            )
            context.summary = summary
            context.last_summary_turn = context.total_messages

    def get_message_count(self, conversation_id: str) -> int:
        if not conversation_id:
            return 0
        with self._lock:
            context = self._store.get(conversation_id)
            return context.total_messages if context else 0

    def get_last_summary_turn(self, conversation_id: str) -> int:
        if not conversation_id:
            return 0
        with self._lock:
            context = self._store.get(conversation_id)
            return context.last_summary_turn if context else 0


_conversation_store = ConversationStore()


def get_conversation_store() -> ConversationStore:
    return _conversation_store


