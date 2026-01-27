from __future__ import annotations

import logging
import sys
import warnings
from typing import Any

# Suppress Pydantic V1 compatibility warning from langsmith on Python 3.14+
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import chat
from .routers import product_chat
from .services.error_handling import (
    build_error_response,
    extract_conversation_id,
    log_exception,
    new_trace_id,
)
from .services.errors import AssistantError, map_exception


def setup_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()
    
    # Determine log level from settings
    log_level = logging.DEBUG if settings.debug else logging.INFO
    
    # Create formatter with detailed output
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Configure root handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    
    # Configure app loggers
    app_logger = logging.getLogger("app")
    app_logger.setLevel(log_level)
    app_logger.handlers.clear()
    app_logger.addHandler(handler)
    app_logger.propagate = False
    
    # Also configure uvicorn access logs to be less verbose
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.setLevel(logging.WARNING)
    
    # Reduce httpx noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


setup_logging()
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

    @app.get("/health", response_model=None)
    async def health() -> dict[str, Any]:
        """Health check endpoint with dependency status."""
        checks = {
            "config_loaded": True,
            "router_config_loaded": False,
            "llm_configured": bool(settings.openai_api_key),
            "langsmith_enabled": bool(settings.langsmith_api_key and settings.langsmith_tracing_v2),
        }
        
        # Check router config
        try:
            from .services.router import get_router_service
            router = get_router_service()
            checks["router_config_loaded"] = len(router._rules) > 0
        except Exception:
            checks["router_config_loaded"] = False
        
        all_ok = checks["config_loaded"] and checks["router_config_loaded"]
        
        return {
            "status": "ok" if all_ok else "degraded",
            "checks": checks,
            "version": settings.assistant_system_prompt_version,
            "environment": settings.env,
        }

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        return await _handle_exception(request, exc, handled=True)

    @app.exception_handler(AssistantError)
    async def assistant_error_handler(request: Request, exc: AssistantError):
        return await _handle_exception(request, exc, handled=True)

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return await _handle_exception(request, exc, handled=True)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return await _handle_exception(request, exc, handled=False)

    app.include_router(chat.router)
    app.include_router(product_chat.router)
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


async def _handle_exception(request: Request, exc: Exception, *, handled: bool):
    trace_id = new_trace_id()
    error_code, http_status, reason = map_exception(exc)
    conversation_id = await extract_conversation_id(request) or trace_id
    debug_payload = {"trace_id": trace_id}
    if isinstance(exc, AssistantError) and getattr(exc, "debug", None):
        debug_payload.update(exc.debug)
    await log_exception(
        request=request,
        exc=exc,
        trace_id=trace_id,
        handled=handled,
        error_code=error_code,
        reason=reason,
        conversation_id=conversation_id,
    )
    return build_error_response(
        conversation_id=conversation_id,
        error_code=error_code,
        reason=reason,
        http_status=http_status,
        debug_payload=debug_payload,
        status_code=http_status,
    )

