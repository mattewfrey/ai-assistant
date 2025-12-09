from __future__ import annotations

import logging
import traceback
import uuid
from typing import Any, Tuple

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

SAFE_ERROR_TEXT = (
    "Кажется, сейчас я не могу обработать ваш запрос. Попробуйте ещё раз чуть позже."
)
SAFE_ERROR_TITLE = "Техническая ошибка"


class AppError(Exception):
    """Base application error for unified handling."""

    def __init__(
        self,
        message: str | None = None,
        *,
        reason: str | None = None,
        http_status: int | None = None,
        debug: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or reason or "")
        self.reason = reason
        self.http_status = http_status
        self.debug = debug or {}


class BadRequestError(AppError):
    """Raised when the request is invalid or cannot be processed."""


class ValidationError(BadRequestError):
    """Raised when the payload fails validation."""


class UpstreamError(AppError):
    """Raised when platform dependencies fail."""


class LLMError(AppError):
    """Raised when LLM / LangChain layer fails."""


class InternalError(AppError):
    """Raised for uncategorized internal failures."""


def _detail_to_reason(detail: Any) -> str:
    if isinstance(detail, dict):
        return detail.get("reason") or detail.get("message") or "unknown"
    if isinstance(detail, list):
        return detail[0] if detail else "unknown"
    if detail:
        return str(detail)
    return "unknown"


def map_exception_to_error_code(exc: Exception) -> Tuple[str, str, int]:
    """Return normalized error code, reason, and HTTP status for the given exception."""

    if isinstance(exc, ValidationError):
        return (
            "BAD_REQUEST",
            exc.reason or "validation_error",
            exc.http_status or status.HTTP_400_BAD_REQUEST,
        )

    if isinstance(exc, BadRequestError):
        return (
            "BAD_REQUEST",
            exc.reason or "bad_request",
            exc.http_status or status.HTTP_400_BAD_REQUEST,
        )

    if isinstance(exc, RequestValidationError):
        return ("BAD_REQUEST", "request_validation_error", status.HTTP_422_UNPROCESSABLE_ENTITY)

    if isinstance(exc, UpstreamError):
        return (
            "UPSTREAM_UNAVAILABLE",
            exc.reason or "upstream_error",
            exc.http_status or status.HTTP_502_BAD_GATEWAY,
        )

    if isinstance(exc, LLMError):
        return (
            "LLM_UNAVAILABLE",
            exc.reason or "llm_error",
            exc.http_status or status.HTTP_502_BAD_GATEWAY,
        )

    if isinstance(exc, HTTPException):
        status_code = exc.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR
        reason = _detail_to_reason(exc.detail)
        if status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            return ("BAD_REQUEST", "rate_limit_exceeded", status_code)
        if status.HTTP_400_BAD_REQUEST <= status_code < status.HTTP_500_INTERNAL_SERVER_ERROR:
            return ("BAD_REQUEST", reason or "bad_request", status_code)
        if status_code in {
            status.HTTP_502_BAD_GATEWAY,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            status.HTTP_504_GATEWAY_TIMEOUT,
        }:
            return ("UPSTREAM_UNAVAILABLE", reason or "upstream_error", status_code)
        return ("INTERNAL_ERROR", reason or "internal_error", status_code)

    if isinstance(exc, InternalError):
        return (
            "INTERNAL_ERROR",
            exc.reason or "internal_error",
            exc.http_status or status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return (
        "INTERNAL_ERROR",
        getattr(exc, "reason", None) or exc.__class__.__name__,
        getattr(exc, "http_status", None) or status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def build_error_response(
    *,
    error_code: str,
    reason: str,
    status_code: int,
    debug_payload: dict[str, Any] | None = None,
) -> JSONResponse:
    meta: dict[str, Any] = {"error": {"code": error_code, "reason": reason}}
    if debug_payload:
        meta["debug"] = debug_payload
    return JSONResponse(
        status_code=status_code,
        content={
            "reply": {"text": SAFE_ERROR_TEXT, "title": SAFE_ERROR_TITLE},
            "data": None,
            "meta": meta,
        },
    )


def new_trace_id() -> str:
    return uuid.uuid4().hex


async def get_request_payload(request: Request) -> str | None:
    try:
        body = await request.body()
        if not body:
            return None
        return body.decode("utf-8", errors="replace")
    except Exception:
        return None


async def log_exception(
    *,
    request: Request,
    exc: Exception,
    trace_id: str,
    handled: bool,
) -> None:
    payload = await get_request_payload(request)
    log_message = "Handled application error" if handled else "Unhandled application error"
    log_method = logger.warning if handled else logger.exception
    log_method(
        "%s trace_id=%s path=%s reason=%s payload=%s",
        log_message,
        trace_id,
        request.url.path,
        getattr(exc, "reason", None) or exc.__class__.__name__,
        payload,
        exc_info=exc if not handled else None,
    )
    if handled:
        logger.debug(
            "Full traceback for trace_id=%s\n%s",
            trace_id,
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )


