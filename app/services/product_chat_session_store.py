from __future__ import annotations

from threading import Lock
from typing import Dict, Optional


class ProductChatSessionStore:
    """In-memory mapping between conversation_id and product_id."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._mapping: Dict[str, str] = {}

    def get_product_id(self, conversation_id: str) -> Optional[str]:
        if not conversation_id:
            return None
        with self._lock:
            return self._mapping.get(conversation_id)

    def set_product_id(self, conversation_id: str, product_id: str) -> None:
        if not conversation_id:
            return
        with self._lock:
            self._mapping[conversation_id] = product_id


_product_chat_session_store = ProductChatSessionStore()


def get_product_chat_session_store() -> ProductChatSessionStore:
    return _product_chat_session_store

