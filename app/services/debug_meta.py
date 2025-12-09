from __future__ import annotations

from typing import Any, Dict, List, Optional


class DebugMetaBuilder:
    """Helper to build a unified meta.debug payload."""

    def __init__(
        self,
        *,
        trace_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self._llm_used: Optional[bool] = None
        self._llm_cached: Optional[bool] = None
        self._router_matched: Optional[bool] = None
        self._slot_filling_used: Optional[bool] = None
        self._intent_chain: List[str] = []
        self._source: Optional[str] = None
        self._trace_id = trace_id
        self._request_id = request_id
        self._pending_slots: Optional[bool] = None
        self._extra: Dict[str, Any] = {}

    def set_llm_used(self, used: bool, *, cached: bool | None = None) -> "DebugMetaBuilder":
        self._llm_used = used
        if cached is not None:
            self._llm_cached = cached
        return self

    def set_llm_cached(self, cached: bool) -> "DebugMetaBuilder":
        self._llm_cached = cached
        return self

    def set_router_matched(self, matched: bool) -> "DebugMetaBuilder":
        self._router_matched = matched
        return self

    def set_slot_filling_used(self, used: bool) -> "DebugMetaBuilder":
        self._slot_filling_used = used
        return self

    def add_intent(self, intent: str | None) -> "DebugMetaBuilder":
        if intent and intent not in self._intent_chain:
            self._intent_chain.append(intent)
        return self

    def set_source(self, source: str) -> "DebugMetaBuilder":
        self._source = source
        return self

    def set_trace_id(self, trace_id: str | None) -> "DebugMetaBuilder":
        if trace_id:
            self._trace_id = trace_id
        return self

    def set_request_id(self, request_id: str | None) -> "DebugMetaBuilder":
        if request_id:
            self._request_id = request_id
        return self

    def set_pending_slots(self, pending: bool) -> "DebugMetaBuilder":
        self._pending_slots = pending
        return self

    def add_extra(self, key: str, value: Any) -> "DebugMetaBuilder":
        self._extra[key] = value
        return self

    def merge_existing(self, debug: Dict[str, Any] | None) -> "DebugMetaBuilder":
        if not debug:
            return self
        self.set_llm_used(debug.get("llm_used")) if debug.get("llm_used") is not None else None
        if "llm_cached" in debug:
            self.set_llm_cached(bool(debug.get("llm_cached")))
        if "router_matched" in debug:
            self.set_router_matched(bool(debug.get("router_matched")))
        if "slot_filling_used" in debug:
            self.set_slot_filling_used(bool(debug.get("slot_filling_used")))
        if debug.get("intent_chain"):
            for intent in debug.get("intent_chain") or []:
                self.add_intent(intent)
        if debug.get("source"):
            self.set_source(str(debug.get("source")))
        if debug.get("trace_id"):
            self.set_trace_id(str(debug.get("trace_id")))
        if debug.get("request_id"):
            self.set_request_id(str(debug.get("request_id")))
        if "pending_slots" in debug:
            self.set_pending_slots(bool(debug.get("pending_slots")))
        for key, value in debug.items():
            if key not in {
                "llm_used",
                "llm_cached",
                "router_matched",
                "slot_filling_used",
                "intent_chain",
                "source",
                "trace_id",
                "request_id",
                "pending_slots",
            }:
                self._extra[key] = value
        return self

    def build(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "llm_used": bool(self._llm_used) if self._llm_used is not None else False,
            "llm_cached": bool(self._llm_cached) if self._llm_cached is not None else False,
            "router_matched": bool(self._router_matched) if self._router_matched is not None else False,
            "slot_filling_used": bool(self._slot_filling_used) if self._slot_filling_used is not None else False,
            "intent_chain": list(self._intent_chain),
            "source": self._source,
        }
        if self._trace_id:
            payload["trace_id"] = self._trace_id
        if self._request_id:
            payload["request_id"] = self._request_id
        if self._pending_slots is not None:
            payload["pending_slots"] = self._pending_slots
        payload.update(self._extra)
        return payload


