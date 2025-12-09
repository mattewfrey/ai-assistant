from __future__ import annotations

import traceback
from typing import Any, Tuple

from fastapi import HTTPException, status
from fastapi.exceptions import RequestValidationError


class AssistantError(Exception):
    code: str = "INTERNAL_ERROR"
    http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(
        self,
        message: str | None = None,
        *,
        reason: str | None = None,
        http_status: int | None = None,
        debug: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or reason or "")
        self.reason = reason or message or self.code
        if http_status is not None:
            self.http_status = http_status
        self.debug = debug or {}


class BadRequestError(AssistantError):
    code = "BAD_REQUEST"
    http_status = status.HTTP_400_BAD_REQUEST


class UpstreamError(AssistantError):
    code = "UPSTREAM_UNAVAILABLE"
    http_status = status.HTTP_502_BAD_GATEWAY


class LLMError(AssistantError):
    code = "LLM_ERROR"
    http_status = status.HTTP_502_BAD_GATEWAY


def _detail_to_reason(detail: Any) -> str:
    if isinstance(detail, dict):
        return detail.get("reason") or detail.get("message") or "unknown"
    if isinstance(detail, list):
        return detail[0] if detail else "unknown"
    if detail:
        return str(detail)
    return "unknown"


def _format_trace(exc: Exception) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()


def _format_validation_error(exc: RequestValidationError) -> str:
    try:
        first_error = exc.errors()[0]
        loc = ".".join(str(part) for part in first_error.get("loc", []) if part is not None)
        msg = first_error.get("msg") or "validation_error"
        return f"{loc}: {msg}" if loc else msg
    except Exception:
        return "validation_error"


def map_exception(exc: Exception) -> Tuple[str, int, str]:
    """
    Возвращает (code, http_status, reason_for_logs)
    """

    if isinstance(exc, AssistantError):
        reason = getattr(exc, "reason", None) or exc.__class__.__name__
        return exc.code, getattr(exc, "http_status", status.HTTP_500_INTERNAL_SERVER_ERROR), reason

    if isinstance(exc, RequestValidationError):
        return (
            "BAD_REQUEST",
            status.HTTP_400_BAD_REQUEST,
            _format_validation_error(exc),
        )

    if isinstance(exc, ValueError):
        return (
            "BAD_REQUEST",
            status.HTTP_400_BAD_REQUEST,
            str(exc) or exc.__class__.__name__,
        )

    if isinstance(exc, HTTPException):
        status_code = exc.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR
        reason = _detail_to_reason(exc.detail)
        if status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            return ("BAD_REQUEST", status_code, reason or "rate_limit_exceeded")
        if status.HTTP_400_BAD_REQUEST <= status_code < status.HTTP_500_INTERNAL_SERVER_ERROR:
            return ("BAD_REQUEST", status_code, reason or "bad_request")
        if status_code in {
            status.HTTP_502_BAD_GATEWAY,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            status.HTTP_504_GATEWAY_TIMEOUT,
        }:
            return ("UPSTREAM_UNAVAILABLE", status_code, reason or "upstream_error")
        return ("INTERNAL_ERROR", status_code, reason or "internal_error")

    return (
        "INTERNAL_ERROR",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        _format_trace(exc),
    )

