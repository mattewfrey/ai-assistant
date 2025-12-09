from __future__ import annotations

import logging
from typing import Any


class RequestLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that prefixes messages with trace/user/conversation context."""

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        trace_id = self.extra.get("trace_id") or "-"
        user_id = self.extra.get("user_id") or "-"
        conversation_id = self.extra.get("conversation_id") or "-"
        prefix = f"trace_id={trace_id} user_id={user_id} conv_id={conversation_id}"
        return f'{prefix} msg="{msg}"', kwargs


def get_request_logger(
    logger: logging.Logger | str,
    *,
    trace_id: str | None,
    user_id: str | None,
    conversation_id: str | None,
) -> RequestLoggerAdapter:
    base_logger = logging.getLogger(logger) if isinstance(logger, str) else logger
    return RequestLoggerAdapter(
        base_logger,
        {
            "trace_id": trace_id or "-",
            "user_id": user_id or "-",
            "conversation_id": conversation_id or "-",
        },
    )

