from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import httpx

from ..config import Settings
from ..utils.logging import get_request_logger
from .errors import UpstreamError

logger = logging.getLogger(__name__)


class ProductGatewayClientError(UpstreamError):
    """Raised when product gateway API fails."""


class ProductGatewayClient:
    """HTTP client for product gateway (product-search)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.product_gateway_base_url.rstrip("/")

    async def fetch_product_full(
        self,
        *,
        product_id: str,
        authorization: Optional[str] = None,
        trace_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch full product data using GET /api/v1/product-search/{id}.
        
        This endpoint returns complete product information including:
        - All attributes (dosage, contraindications, side effects, etc.)
        - Metadata with breadcrumbs/categories
        - Abstract product info (variants)
        - Active ingredient and manufacturer info
        """
        if not product_id:
            raise ProductGatewayClientError("product_id is required")

        # Use GET /api/v1/product-search/{id} for full product data
        url = f"{self._base_url}/api/v1/product-search/{product_id}"
        headers: Dict[str, str] = {"Flex-Locale": "country=RU;bs=gz.ru"}
        auth_header = self._resolve_auth_header(authorization)
        if auth_header:
            headers["Authorization"] = auth_header
        if trace_id:
            headers["X-Request-Id"] = trace_id

        timeout = httpx.Timeout(self._settings.http_timeout_seconds)
        req_logger = get_request_logger(
            logger,
            trace_id=trace_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(url, headers=headers)
                elapsed_ms = (time.perf_counter() - start) * 1000
                req_logger.info(
                    "product_gateway.fetch_product_full product_id=%s status=%s latency_ms=%.1f",
                    product_id,
                    response.status_code,
                    elapsed_ms,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:
                req_logger.error("product_gateway error url=%s error=%s", url, exc)
                raise ProductGatewayClientError(str(exc)) from exc

    def _resolve_auth_header(self, authorization: Optional[str]) -> Optional[str]:
        if authorization:
            return authorization
        token = self._settings.product_gateway_token
        if not token:
            return None
        normalized = token.strip()
        lower = normalized.lower()
        if lower.startswith("bearer ") or lower.startswith("token "):
            return normalized
        return f"Bearer {normalized}"

