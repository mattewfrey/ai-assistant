from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List


@dataclass
class MetricsSnapshot:
    llm_calls_total: int
    llm_calls_per_user: Dict[str, int]
    llm_cache_hits: int
    router_matches: int
    router_misses: int
    router_hit_rate: float
    slot_success: int
    slot_prompts: int
    slot_filling_success_rate: float
    avg_tokens_per_call: float
    # New metrics
    beautify_calls: int = 0
    beautify_skipped: int = 0
    beautify_cache_hits: int = 0
    safety_filter_triggers: int = 0
    rate_limit_rejections: int = 0
    avg_response_latency_ms: float = 0.0


@dataclass
class RateLimitWindow:
    """Sliding window for rate limiting."""
    timestamps: List[float] = field(default_factory=list)


class MetricsService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._llm_calls_total = 0
        self._llm_calls_per_user: Dict[str, int] = {}
        self._llm_cache_hits = 0
        self._llm_tokens_total = 0
        self._router_matches = 0
        self._router_misses = 0
        self._slot_success = 0
        self._slot_prompts = 0
        # New metrics
        self._beautify_calls = 0
        self._beautify_skipped = 0
        self._beautify_cache_hits = 0
        self._safety_filter_triggers = 0
        self._rate_limit_rejections = 0
        self._response_latencies: List[float] = []
        self._max_latency_samples = 1000
        # Rate limiting state
        self._rate_limit_windows: Dict[str, RateLimitWindow] = defaultdict(RateLimitWindow)

    def record_llm_call(
        self,
        *,
        user_id: str | None,
        cached: bool,
        token_usage: Dict[str, int] | None,
    ) -> None:
        with self._lock:
            if cached:
                self._llm_cache_hits += 1
                return
            self._llm_calls_total += 1
            if user_id:
                self._llm_calls_per_user[user_id] = self._llm_calls_per_user.get(user_id, 0) + 1
            tokens = 0
            if token_usage:
                tokens = int(token_usage.get("total_tokens", 0) or 0)
            self._llm_tokens_total += tokens

    def record_router_match(self, matched: bool) -> None:
        with self._lock:
            if matched:
                self._router_matches += 1
            else:
                self._router_misses += 1

    def record_slot_prompt(self) -> None:
        with self._lock:
            self._slot_prompts += 1

    def record_slot_success(self) -> None:
        with self._lock:
            self._slot_success += 1

    def record_beautify_call(self, *, cached: bool = False) -> None:
        """Record a beautify_reply call."""
        with self._lock:
            if cached:
                self._beautify_cache_hits += 1
            else:
                self._beautify_calls += 1

    def record_beautify_skipped(self) -> None:
        """Record when beautify_reply was skipped (no API key, etc.)."""
        with self._lock:
            self._beautify_skipped += 1

    def record_safety_filter_trigger(self) -> None:
        """Record when SafetyFilter replaced unsafe content."""
        with self._lock:
            self._safety_filter_triggers += 1

    def record_rate_limit_rejection(self) -> None:
        """Record when a request was rejected due to rate limiting."""
        with self._lock:
            self._rate_limit_rejections += 1

    def record_response_latency(self, latency_ms: float) -> None:
        """Record response latency in milliseconds."""
        with self._lock:
            self._response_latencies.append(latency_ms)
            # Keep only recent samples
            if len(self._response_latencies) > self._max_latency_samples:
                self._response_latencies = self._response_latencies[-self._max_latency_samples:]

    def check_rate_limit(
        self,
        user_id: str | None,
        window_seconds: int = 60,
        max_calls: int = 20,
    ) -> bool:
        """
        Check if user is within rate limit using sliding window.
        Returns True if request is allowed, False if rate limited.
        """
        if not user_id:
            # Anonymous users get a shared bucket with stricter limits
            user_id = "__anonymous__"
            max_calls = max(5, max_calls // 4)

        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            window = self._rate_limit_windows[user_id]
            # Remove expired timestamps
            window.timestamps = [ts for ts in window.timestamps if ts > cutoff]
            
            if len(window.timestamps) >= max_calls:
                self._rate_limit_rejections += 1
                return False
            
            window.timestamps.append(now)
            return True

    def get_user_llm_calls_in_window(
        self,
        user_id: str | None,
        window_seconds: int = 60,
    ) -> int:
        """Get number of LLM calls for user in the current window."""
        if not user_id:
            user_id = "__anonymous__"
        
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            window = self._rate_limit_windows.get(user_id)
            if not window:
                return 0
            return len([ts for ts in window.timestamps if ts > cutoff])

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            router_total = self._router_matches + self._router_misses
            router_hit_rate = (self._router_matches / router_total) if router_total else 0.0
            slot_total = self._slot_success + self._slot_prompts
            slot_rate = (self._slot_success / slot_total) if slot_total else 0.0
            avg_tokens = (self._llm_tokens_total / self._llm_calls_total) if self._llm_calls_total else 0.0
            avg_latency = (
                sum(self._response_latencies) / len(self._response_latencies)
                if self._response_latencies else 0.0
            )
            return MetricsSnapshot(
                llm_calls_total=self._llm_calls_total,
                llm_calls_per_user=dict(self._llm_calls_per_user),
                llm_cache_hits=self._llm_cache_hits,
                router_matches=self._router_matches,
                router_misses=self._router_misses,
                router_hit_rate=router_hit_rate,
                slot_success=self._slot_success,
                slot_prompts=self._slot_prompts,
                slot_filling_success_rate=slot_rate,
                avg_tokens_per_call=avg_tokens,
                beautify_calls=self._beautify_calls,
                beautify_skipped=self._beautify_skipped,
                beautify_cache_hits=self._beautify_cache_hits,
                safety_filter_triggers=self._safety_filter_triggers,
                rate_limit_rejections=self._rate_limit_rejections,
                avg_response_latency_ms=avg_latency,
            )


_metrics_service = MetricsService()


def get_metrics_service() -> MetricsService:
    return _metrics_service
