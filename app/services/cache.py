from __future__ import annotations

import hashlib
import json
import time
from threading import Lock
from typing import Any, Dict, Tuple


class TTLCache:
    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at < now:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        expires_at = time.time() + ttl_seconds
        with self._lock:
            self._store[key] = (expires_at, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        now = time.time()
        removed = 0
        with self._lock:
            expired_keys = [
                key for key, (expires_at, _) in self._store.items()
                if expires_at < now
            ]
            for key in expired_keys:
                del self._store[key]
                removed += 1
        return removed

    def size(self) -> int:
        """Return current cache size."""
        with self._lock:
            return len(self._store)


class CachingService:
    """Namespaced helper for caching LLM replies and platform lookups."""

    def __init__(self, cache: TTLCache | None = None) -> None:
        self._cache = cache or TTLCache()

    def get_llm_response(self, normalized_message: str, profile_signature: str | None) -> Any | None:
        cache_key = self._llm_key(normalized_message, profile_signature)
        return self._cache.get(cache_key)

    def set_llm_response(
        self,
        normalized_message: str,
        profile_signature: str | None,
        payload: Any,
        ttl_seconds: int = 900,
    ) -> None:
        cache_key = self._llm_key(normalized_message, profile_signature)
        self._cache.set(cache_key, payload, ttl_seconds)

    def get_product_results(self, slots: Dict[str, Any]) -> Any | None:
        cache_key = self._product_key(slots)
        return self._cache.get(cache_key)

    def set_product_results(self, slots: Dict[str, Any], payload: Any, ttl_seconds: int = 600) -> None:
        cache_key = self._product_key(slots)
        self._cache.set(cache_key, payload, ttl_seconds)

    def get_beautify_response(
        self,
        base_reply_text: str,
        data_hash: str,
        constraints_hash: str | None = None,
    ) -> Any | None:
        """Get cached beautify_reply response."""
        cache_key = self._beautify_key(base_reply_text, data_hash, constraints_hash)
        return self._cache.get(cache_key)

    def set_beautify_response(
        self,
        base_reply_text: str,
        data_hash: str,
        constraints_hash: str | None,
        payload: Any,
        ttl_seconds: int = 600,
    ) -> None:
        """Cache beautify_reply response."""
        cache_key = self._beautify_key(base_reply_text, data_hash, constraints_hash)
        self._cache.set(cache_key, payload, ttl_seconds)

    def _llm_key(self, normalized_message: str, profile_signature: str | None) -> str:
        signature = profile_signature or "-"
        return f"llm:{normalized_message.strip().lower()}::{signature}"

    def _product_key(self, slots: Dict[str, Any]) -> str:
        normalized = json.dumps(slots or {}, ensure_ascii=False, sort_keys=True)
        return f"search:{normalized}"

    def _beautify_key(
        self,
        base_reply_text: str,
        data_hash: str,
        constraints_hash: str | None,
    ) -> str:
        """Generate cache key for beautify_reply."""
        text_hash = hashlib.md5(base_reply_text.encode()).hexdigest()[:12]
        constraints_part = constraints_hash[:8] if constraints_hash else "-"
        return f"beautify:{text_hash}:{data_hash[:12]}:{constraints_part}"

    @staticmethod
    def compute_data_hash(data: Dict[str, Any]) -> str:
        """Compute hash of data payload for caching."""
        # Only hash essential fields that affect the reply
        essential = {
            "products_count": len(data.get("products", [])),
            "has_cart": bool(data.get("cart")),
            "orders_count": len(data.get("orders", [])),
            "has_pharmacies": bool(data.get("pharmacies")),
        }
        # Add first product IDs for uniqueness
        products = data.get("products", [])[:3]
        essential["product_ids"] = [p.get("id") for p in products if p.get("id")]
        
        serialized = json.dumps(essential, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    @staticmethod
    def compute_constraints_hash(constraints: Dict[str, Any] | None) -> str | None:
        """Compute hash of constraints for caching."""
        if not constraints:
            return None
        # Exclude user_message as it's usually unique
        cacheable = {k: v for k, v in constraints.items() if k != "user_message"}
        if not cacheable:
            return None
        serialized = json.dumps(cacheable, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()


_cache = TTLCache()
_caching_service = CachingService(_cache)


def get_cache() -> TTLCache:
    return _cache


def get_caching_service() -> CachingService:
    return _caching_service
