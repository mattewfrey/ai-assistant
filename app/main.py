from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import chat
from .services.error_handling import (
    AppError,
    build_error_response,
    log_exception,
    map_exception_to_error_code,
    new_trace_id,
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Pharmacy Assistant Gateway",
        version=settings.assistant_system_prompt_version,
        docs_url="/docs" if settings.debug else None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        trace_id = new_trace_id()
        await log_exception(request=request, exc=exc, trace_id=trace_id, handled=True)
        error_code, reason, status_code = map_exception_to_error_code(exc)
        return build_error_response(
            error_code=error_code,
            reason=reason,
            status_code=status_code,
            debug_payload={"trace_id": trace_id},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        trace_id = new_trace_id()
        await log_exception(request=request, exc=exc, trace_id=trace_id, handled=True)
        error_code, reason, status_code = map_exception_to_error_code(exc)
        return build_error_response(
            error_code=error_code,
            reason=reason,
            status_code=status_code,
            debug_payload={"trace_id": trace_id},
        )

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        trace_id = new_trace_id()
        await log_exception(request=request, exc=exc, trace_id=trace_id, handled=True)
        error_code, reason, status_code = map_exception_to_error_code(exc)
        debug_payload = {"trace_id": trace_id}
        if exc.debug:
            debug_payload.update(exc.debug)
        return build_error_response(
            error_code=error_code,
            reason=reason,
            status_code=status_code,
            debug_payload=debug_payload,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        trace_id = new_trace_id()
        await log_exception(request=request, exc=exc, trace_id=trace_id, handled=False)
        error_code, reason, status_code = map_exception_to_error_code(exc)
        return build_error_response(
            error_code=error_code,
            reason=reason,
            status_code=status_code,
            debug_payload={"trace_id": trace_id},
        )

    app.include_router(chat.router)
    if settings.langsmith_api_key and settings.langsmith_tracing_v2:
        logger.info(
            "LangSmith tracing enabled for project=%s",
            settings.langsmith_project or "smart-pharmacy-assistant",
        )
    else:
        logger.info("LangSmith tracing disabled (no API key or flag)")
    logger.info("FastAPI app initialized (env=%s)", settings.env)
    return app


app = create_app()

