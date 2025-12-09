from __future__ import annotations

import logging
from typing import Any


def _format_context(
    *,
    trace_id: str | None,
    user_id: str | None,
    conversation_id: str | None,
    intent: Any = None,
) -> str:
    return (
        f"trace_id={trace_id or '-'} "
        f"user_id={user_id or '-'} "
        f"conversation_id={conversation_id or '-'} "
        f"intent={getattr(intent, 'value', intent) or '-'}"
    )


def log_info(logger: logging.Logger, message: str, *, trace_id=None, user_id=None, conversation_id=None, intent=None) -> None:
    logger.info("%s %s", _format_context(trace_id=trace_id, user_id=user_id, conversation_id=conversation_id, intent=intent), message)


def log_warning(logger: logging.Logger, message: str, *, trace_id=None, user_id=None, conversation_id=None, intent=None) -> None:
    logger.warning("%s %s", _format_context(trace_id=trace_id, user_id=user_id, conversation_id=conversation_id, intent=intent), message)


def log_error(
    logger: logging.Logger,
    message: str,
    *,
    trace_id=None,
    user_id=None,
    conversation_id=None,
    intent=None,
    exc_info: bool | Exception = False,
) -> None:
    logger.error(
        "%s %s",
        _format_context(trace_id=trace_id, user_id=user_id, conversation_id=conversation_id, intent=intent),
        message,
        exc_info=exc_info,
    )


def log_debug(logger: logging.Logger, message: str, *, trace_id=None, user_id=None, conversation_id=None, intent=None) -> None:
    logger.debug("%s %s", _format_context(trace_id=trace_id, user_id=user_id, conversation_id=conversation_id, intent=intent), message)

