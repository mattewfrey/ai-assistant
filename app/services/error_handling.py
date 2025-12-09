from __future__ import annotations

import json
import logging
import traceback
import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from ..utils.logging import get_request_logger

logger = logging.getLogger(__name__)

SAFE_ERROR_TEXT = (
    "Кажется, сейчас я не могу обработать ваш запрос. Попробуйте позже."
)
SAFE_ERROR_TITLE = "Техническая ошибка"


async def get_request_payload(request: Request) -> str | None:
    try:
        body = await request.body()
        if not body:
            return None
        return body.decode("utf-8", errors="replace")
    except Exception:
        return None


async def extract_conversation_id(request: Request) -> str | None:
    payload = await get_request_payload(request)
    if not payload:
        return None
    try:
        body = json.loads(payload)
    except Exception:
        return None
    conversation_id = body.get("conversation_id")
    if isinstance(conversation_id, str) and conversation_id.strip():
        return conversation_id
    return None


def build_error_response(
    *,
    conversation_id: str,
    error_code: str,
    reason: str,
    http_status: int,
    debug_payload: dict[str, Any] | None = None,
    status_code: int | None = None,
) -> JSONResponse:
    meta: dict[str, Any] = {
        "error": {"code": error_code, "reason": reason, "http_status": http_status}
    }
    if debug_payload:
        meta["debug"] = debug_payload
    return JSONResponse(
        status_code=status_code or http_status,
        content={
            "conversation_id": conversation_id,
            "reply": {"text": SAFE_ERROR_TEXT, "title": SAFE_ERROR_TITLE},
            "actions": [],
            "meta": meta,
            "data": None,
        },
    )


def new_trace_id() -> str:
    return uuid.uuid4().hex


async def log_exception(
    *,
    request: Request,
    exc: Exception,
    trace_id: str,
    handled: bool,
    error_code: str,
    reason: str,
    conversation_id: str | None = None,
) -> None:
    payload = await get_request_payload(request)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_message = "Handled application error" if handled else "Unhandled application error"
    req_logger = get_request_logger(logger, trace_id=trace_id, user_id=None, conversation_id=conversation_id)
    log_method = req_logger.warning if handled else req_logger.exception
    log_method(
        '%s code=%s error_type=%s path=%s reason=%s payload=%s',
        log_message,
        error_code,
        type(exc).__name__,
        request.url.path,
        reason,
        payload,
        exc_info=exc if not handled else None,
    )
    req_logger.debug("Full traceback\n%s", tb)

