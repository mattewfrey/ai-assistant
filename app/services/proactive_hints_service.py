"""Service for generating proactive chat hints based on user behavior and product context."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..models.proactive_hints import (
    ProactiveHint,
    ProactiveHintsRequest,
    ProactiveHintsResponse,
    ProactiveHintType,
    ProactiveTriggerType,
)
from .product_context_builder import ProductContextBuilder

logger = logging.getLogger(__name__)


class ProactiveHintsService:
    """Service for generating context-aware proactive hints."""

    def __init__(
        self,
        *,
        settings: Settings,
        context_builder: ProductContextBuilder,
    ) -> None:
        self._settings = settings
        self._context_builder = context_builder

    async def get_hints(
        self,
        *,
        request: ProactiveHintsRequest,
        authorization: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> ProactiveHintsResponse:
        """Get proactive hints for a product based on trigger context."""

        # Get product context
        context_result = await self._context_builder.get_context(
            product_id=request.product_id,
            store_id=request.store_id,
            shipping_method=None,
            authorization=authorization,
            trace_id=trace_id,
            conversation_id=None,
            user_id=request.user_id,
        )
        context = context_result.context

        # Generate hints based on trigger type and context
        hints = self._generate_hints(
            trigger_type=request.trigger_type,
            context=context,
            limit=request.limit,
        )

        logger.info(
            "proactive_hints.generated product_id=%s trigger=%s hints_count=%d",
            request.product_id,
            request.trigger_type.value,
            len(hints),
        )

        return ProactiveHintsResponse(
            product_id=request.product_id,
            hints=hints,
            trigger_type=request.trigger_type,
            meta={
                "context_cache_hit": context_result.cache_hit,
                "trigger_type": request.trigger_type.value,
            },
        )

    def _generate_hints(
        self,
        trigger_type: ProactiveTriggerType,
        context: Dict[str, Any],
        limit: int,
    ) -> List[ProactiveHint]:
        """Generate hints based on trigger type and product context."""
        hints: List[ProactiveHint] = []
        product = context.get("product", {})
        pricing = context.get("pricing", {})
        availability = context.get("availability", {})

        product_name = product.get("name", "этот товар")
        short_name = product_name[:30] + "..." if len(product_name) > 30 else product_name

        # Trigger-specific hints
        if trigger_type == ProactiveTriggerType.TIME_ON_PAGE:
            hints.extend(self._hints_for_time_on_page(product, pricing, availability, short_name))

        elif trigger_type == ProactiveTriggerType.SCROLL_DEPTH:
            hints.extend(self._hints_for_scroll_depth(product, pricing, availability, short_name))

        elif trigger_type == ProactiveTriggerType.EXIT_INTENT:
            hints.extend(self._hints_for_exit_intent(product, pricing, availability, short_name))

        elif trigger_type == ProactiveTriggerType.IDLE:
            hints.extend(self._hints_for_idle(product, pricing, availability, short_name))

        elif trigger_type == ProactiveTriggerType.RETURN_VISIT:
            hints.extend(self._hints_for_return_visit(product, pricing, availability, short_name))

        elif trigger_type == ProactiveTriggerType.CART_HESITATION:
            hints.extend(self._hints_for_cart_hesitation(product, pricing, availability, short_name))

        # Sort by priority and limit
        hints.sort(key=lambda h: h.priority, reverse=True)
        return hints[:limit]

    def _hints_for_time_on_page(
        self, product: Dict, pricing: Dict, availability: Dict, short_name: str
    ) -> List[ProactiveHint]:
        """Hints when user has been on page for a while."""
        hints = []

        # Suggest asking about the product
        hints.append(ProactiveHint(
            hint_type=ProactiveHintType.HELP_OFFER,
            message=f"Изучаете {short_name}? Могу помочь с вопросами!",
            suggested_question="Расскажи подробнее о товаре",
            priority=7,
            action_label="Задать вопрос",
        ))

        # If has price with discount
        prices = pricing.get("prices", [])
        if prices:
            price_info = prices[0]
            if price_info.get("price_no_discount") and price_info.get("price"):
                if price_info["price_no_discount"] > price_info["price"]:
                    hints.append(ProactiveHint(
                        hint_type=ProactiveHintType.PRICE_INFO,
                        message=f"Сейчас действует скидка! Цена {price_info['price']}₽ вместо {price_info['price_no_discount']}₽",
                        suggested_question="Сколько стоит?",
                        priority=9,
                    ))

        return hints

    def _hints_for_scroll_depth(
        self, product: Dict, pricing: Dict, availability: Dict, short_name: str
    ) -> List[ProactiveHint]:
        """Hints when user scrolled deep into page."""
        hints = []

        # User is reading carefully - suggest detailed questions
        hints.append(ProactiveHint(
            hint_type=ProactiveHintType.FAQ_SUGGESTION,
            message="Частый вопрос: Какая форма выпуска и состав?",
            suggested_question="Какой состав и форма выпуска?",
            priority=6,
            action_label="Спросить",
        ))

        # Delivery info if available
        if product.get("delivery_available"):
            hints.append(ProactiveHint(
                hint_type=ProactiveHintType.DELIVERY_INFO,
                message="Узнайте о вариантах доставки",
                suggested_question="Какие варианты доставки?",
                priority=5,
            ))

        return hints

    def _hints_for_exit_intent(
        self, product: Dict, pricing: Dict, availability: Dict, short_name: str
    ) -> List[ProactiveHint]:
        """Hints when user is about to leave - highest priority."""
        hints = []

        # Most important - availability
        pickup_count = availability.get("pickup_stores_count")
        if pickup_count and pickup_count > 0:
            hints.append(ProactiveHint(
                hint_type=ProactiveHintType.AVAILABILITY_ALERT,
                message=f"Товар есть в {pickup_count} аптеках рядом. Нужна помощь с выбором?",
                suggested_question="Где можно забрать?",
                priority=10,
                action_label="Узнать наличие",
            ))
        else:
            # Generic exit intent
            hints.append(ProactiveHint(
                hint_type=ProactiveHintType.HELP_OFFER,
                message="Остались вопросы? Я могу помочь!",
                suggested_question="Есть вопрос",
                priority=8,
                action_label="Спросить",
            ))

        # Price reminder
        prices = pricing.get("prices", [])
        if prices:
            price = prices[0].get("price")
            if price:
                hints.append(ProactiveHint(
                    hint_type=ProactiveHintType.PRICE_INFO,
                    message=f"Цена {price}₽. Есть вопросы?",
                    suggested_question="Сколько стоит?",
                    priority=7,
                ))

        return hints

    def _hints_for_idle(
        self, product: Dict, pricing: Dict, availability: Dict, short_name: str
    ) -> List[ProactiveHint]:
        """Hints when user has been idle."""
        hints = []

        hints.append(ProactiveHint(
            hint_type=ProactiveHintType.HELP_OFFER,
            message="Нужна помощь с выбором? Задайте вопрос!",
            suggested_question="Помоги с выбором",
            priority=5,
        ))

        return hints

    def _hints_for_return_visit(
        self, product: Dict, pricing: Dict, availability: Dict, short_name: str
    ) -> List[ProactiveHint]:
        """Hints when user returned to the same product."""
        hints = []

        hints.append(ProactiveHint(
            hint_type=ProactiveHintType.HELP_OFFER,
            message=f"Снова смотрите {short_name}? Могу ответить на вопросы!",
            suggested_question="У меня есть вопрос",
            priority=8,
            action_label="Задать вопрос",
        ))

        # Check availability for returning user
        pickup_count = availability.get("pickup_stores_count")
        if pickup_count and pickup_count > 0:
            hints.append(ProactiveHint(
                hint_type=ProactiveHintType.AVAILABILITY_ALERT,
                message=f"Товар в наличии в {pickup_count} аптеках",
                suggested_question="Есть в наличии?",
                priority=7,
            ))

        return hints

    def _hints_for_cart_hesitation(
        self, product: Dict, pricing: Dict, availability: Dict, short_name: str
    ) -> List[ProactiveHint]:
        """Hints when user has product in cart but hesitates."""
        hints = []

        # Prescription info for pharma
        if product.get("prescription") is False:
            hints.append(ProactiveHint(
                hint_type=ProactiveHintType.FAQ_SUGGESTION,
                message="Рецепт не требуется. Есть вопросы по применению?",
                suggested_question="Как принимать?",
                priority=9,
            ))

        # Delivery info
        if product.get("delivery_available"):
            hints.append(ProactiveHint(
                hint_type=ProactiveHintType.DELIVERY_INFO,
                message="Доступна доставка. Узнайте сроки!",
                suggested_question="Когда доставите?",
                priority=8,
            ))

        return hints
