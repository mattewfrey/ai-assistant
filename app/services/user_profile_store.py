from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List

from ..models import UserPreferences, UserProfile


class UserProfileStore:
    """In-memory cache for per-user personalization data.

    TODO: replace with a persistent storage (DB/cache) shared across instances.
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, UserProfile] = {}
        self._lock = Lock()

    def get_or_create(self, user_id: str) -> UserProfile:
        if not user_id:
            raise ValueError("user_id must be provided to load a profile")
        with self._lock:
            profile = self._profiles.get(user_id)
            if profile is None:
                profile = UserProfile(user_id=user_id)
                self._profiles[user_id] = profile
            return profile

    def update_preferences(self, user_id: str, **kwargs: Any) -> UserProfile:
        """Merge preference updates into the profile."""

        if not user_id:
            raise ValueError("user_id must be provided to update preferences")

        FIELD_ALIASES = {
            "price_ceiling": "default_max_price",
            "preferred_dosage_forms": "preferred_forms",
        }
        with self._lock:
            profile = self._profiles.setdefault(user_id, UserProfile(user_id=user_id))
            preferences = profile.preferences
            pref_data = preferences.model_dump()
            updated = False

            for key, value in kwargs.items():
                normalized_key = FIELD_ALIASES.get(key, key)
                if normalized_key not in preferences.model_fields:
                    continue
                key = normalized_key
                if key in {"preferred_forms"}:
                    if not value:
                        continue
                    raw_iterable = value
                    if isinstance(value, str):
                        raw_iterable = [value]
                    elif not isinstance(value, (list, tuple, set)):
                        continue
                    existing = pref_data.get(key) or []
                    additions = [str(form).strip().lower() for form in raw_iterable if str(form).strip()]
                    merged = self._merge_unique(existing, additions)
                    if merged != existing:
                        pref_data[key] = merged
                        updated = True
                    continue
                if key == "default_max_price":
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                if key == "age":
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        continue
                if value is None:
                    continue
                if pref_data.get(key) != value:
                    pref_data[key] = value
                    updated = True

            if updated:
                profile.preferences = UserPreferences(**pref_data)
                self._profiles[user_id] = profile
            return profile

    def add_tag(self, user_id: str, tag: str) -> UserProfile:
        if not user_id:
            raise ValueError("user_id must be provided to add a tag")
        normalized = tag.strip().lower()
        with self._lock:
            profile = self._profiles.setdefault(user_id, UserProfile(user_id=user_id))
            tags = profile.tags or []
            if normalized and normalized not in tags:
                tags.append(normalized)
                profile.tags = tags
                self._profiles[user_id] = profile
            return profile

    def _merge_unique(self, existing: List[str], new_items: List[str]) -> List[str]:
        seen = {item for item in existing}
        merged = existing.copy()
        for item in new_items:
            if not item:
                continue
            if item not in seen:
                merged.append(item)
                seen.add(item)
        return merged


_user_profile_store = UserProfileStore()


def get_user_profile_store() -> UserProfileStore:
    return _user_profile_store


