from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..models.product_faq import ProductFAQItem, ProductFAQLLMResult, ProductFAQResponse
from .cache import CachingService, get_caching_service
from .product_context_builder import ProductContextBuilder

logger = logging.getLogger(__name__)

# Default TTL for FAQ cache (1 hour)
FAQ_CACHE_TTL_SECONDS = 3600


@dataclass
class FAQGenerationResult:
    """Result of FAQ generation from LLM."""

    result: ProductFAQLLMResult
    token_usage: Dict[str, Any]
    cached: bool


class ProductFAQService:
    """Service for generating and caching product FAQs."""

    def __init__(
        self,
        *,
        settings: Settings,
        context_builder: ProductContextBuilder,
        llm_client: Any | None,
        cache: CachingService | None = None,
    ) -> None:
        self._settings = settings
        self._context_builder = context_builder
        self._llm_client = llm_client
        self._cache = cache or get_caching_service()

    async def get_faqs(
        self,
        *,
        product_id: str,
        store_id: Optional[str] = None,
        shipping_method: Optional[str] = None,
        authorization: Optional[str] = None,
        trace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        force_refresh: bool = False,
    ) -> ProductFAQResponse:
        """Get FAQs for a product, using cache if available."""

        cache_key = self._faq_cache_key(product_id, store_id, shipping_method)

        # Check cache first
        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached:
                logger.info("product_faq.cache_hit product_id=%s", product_id)
                return ProductFAQResponse(
                    product_id=product_id,
                    product_name=cached.get("product_name"),
                    faqs=[ProductFAQItem(**item) for item in cached.get("faqs", [])],
                    generated_at=cached.get("generated_at"),
                    cache_hit=True,
                    meta={"source": "cache"},
                )

        # Get product context
        context_result = await self._context_builder.get_context(
            product_id=product_id,
            store_id=store_id,
            shipping_method=shipping_method,
            authorization=authorization,
            trace_id=trace_id,
            conversation_id=None,
            user_id=user_id,
        )
        context = context_result.context
        product_name = context.get("product", {}).get("name")

        # Generate FAQs using LLM or fallback to rule-based
        if self._llm_client is not None:
            faqs = await self._generate_faqs_with_llm(product_id, context, trace_id)
        else:
            faqs = self._generate_fallback_faqs(context)

        # Sort by priority (descending)
        faqs.sort(key=lambda x: x.priority, reverse=True)

        # Cache the result
        generated_at = datetime.now(timezone.utc).isoformat()
        cache_payload = {
            "product_name": product_name,
            "faqs": [faq.model_dump() for faq in faqs],
            "generated_at": generated_at,
        }
        self._cache.set(cache_key, cache_payload, FAQ_CACHE_TTL_SECONDS)

        logger.info(
            "product_faq.generated product_id=%s faq_count=%d",
            product_id,
            len(faqs),
        )

        return ProductFAQResponse(
            product_id=product_id,
            product_name=product_name,
            faqs=faqs,
            generated_at=generated_at,
            cache_hit=False,
            meta={
                "source": "llm" if self._llm_client else "fallback",
                "context_cache_hit": context_result.cache_hit,
            },
        )

    async def _generate_faqs_with_llm(
        self,
        product_id: str,
        context: Dict[str, Any],
        trace_id: Optional[str],
    ) -> List[ProductFAQItem]:
        """Generate FAQs using LLM."""
        try:
            result = await self._llm_client.generate_product_faqs(
                product_id=product_id,
                context_json=context,
                trace_id=trace_id,
            )
            return list(result.result.faqs)
        except Exception as exc:
            logger.warning("FAQ LLM generation failed: %s, using fallback", exc)
            return self._generate_fallback_faqs(context)

    def _generate_fallback_faqs(self, context: Dict[str, Any]) -> List[ProductFAQItem]:
        """Generate basic FAQs using rule-based logic when LLM is unavailable."""
        faqs: List[ProductFAQItem] = []
        product = context.get("product", {})
        pricing = context.get("pricing", {})
        availability = context.get("availability", {})
        attributes = context.get("attributes", [])

        product_name = product.get("name", "товар")

        # Price FAQ
        prices = pricing.get("prices", [])
        if prices:
            price_info = prices[0]
            price = price_info.get("price")
            price_no_discount = price_info.get("price_no_discount")
            if price:
                if price_no_discount and price_no_discount > price:
                    discount = round((1 - price / price_no_discount) * 100)
                    answer = f"Цена {price}₽ (скидка {discount}%, было {price_no_discount}₽)."
                else:
                    answer = f"Цена {price}₽."
                faqs.append(ProductFAQItem(
                    question="Сколько стоит?",
                    answer=answer,
                    category="price",
                    priority=10,
                    used_fields=["pricing.prices"],
                ))

        # Availability FAQ
        stocks = availability.get("stocks", [])
        pickup_count = availability.get("pickup_stores_count")
        if stocks or pickup_count:
            if pickup_count and pickup_count > 0:
                answer = f"Да, доступен в {pickup_count} аптеках для самовывоза."
            elif stocks:
                answer = "Да, есть в наличии."
            else:
                answer = "Уточните наличие по телефону."
            faqs.append(ProductFAQItem(
                question="Есть в наличии?",
                answer=answer,
                category="availability",
                priority=10,
                used_fields=["availability.stocks", "availability.pickup_stores_count"],
            ))

        # Delivery FAQ
        delivery_available = product.get("delivery_available")
        ship_to_store = availability.get("ship_to_store")
        if delivery_available is not None or ship_to_store is not None:
            if delivery_available:
                answer = "Да, доступна доставка курьером."
            elif ship_to_store:
                answer = "Доставка в аптеку доступна."
            else:
                answer = "Только самовывоз из аптек."
            faqs.append(ProductFAQItem(
                question="Есть доставка?",
                answer=answer,
                category="availability",
                priority=8,
                used_fields=["product.delivery_available", "availability.ship_to_store"],
            ))

        # Prescription FAQ
        prescription = product.get("prescription")
        if prescription is True:
            faqs.append(ProductFAQItem(
                question="Нужен ли рецепт?",
                answer="Да, препарат отпускается строго по рецепту врача.",
                category="general",
                priority=9,
                used_fields=["product.prescription"],
            ))
        elif prescription is False:
            faqs.append(ProductFAQItem(
                question="Нужен ли рецепт?",
                answer="Нет, препарат продаётся без рецепта.",
                category="general",
                priority=7,
                used_fields=["product.prescription"],
            ))

        # Manufacturer FAQ
        manufacturer = product.get("manufacturer")
        country = product.get("country")
        if manufacturer or country:
            parts = []
            if manufacturer:
                parts.append(f"производитель {manufacturer}")
            if country:
                parts.append(f"страна {country}")
            answer = ", ".join(parts).capitalize() + "."
            faqs.append(ProductFAQItem(
                question="Кто производитель?",
                answer=answer,
                category="general",
                priority=4,
                used_fields=["product.manufacturer", "product.country"],
            ))

        # Composition/Form from attributes
        form_attr = next((a for a in attributes if a.get("code") in ("form", "release_form", "forma_vypuska")), None)
        if form_attr:
            faqs.append(ProductFAQItem(
                question="Какая форма выпуска?",
                answer=f"{form_attr.get('value', 'Не указано')}.",
                category="composition",
                priority=6,
                used_fields=[f"attributes[code={form_attr.get('code')}]"],
            ))

        return faqs

    @staticmethod
    def _faq_cache_key(product_id: str, store_id: Optional[str], shipping_method: Optional[str]) -> str:
        store_part = store_id or "-"
        ship_part = shipping_method or "-"
        return f"product_faq:{product_id}:{store_part}:{ship_part}"
