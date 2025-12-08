from __future__ import annotations

import logging
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Sequence

import httpx
from pydantic import BaseModel, Field, ValidationError

from ..config import Settings
from ..intents import IntentType
from ..models import AssistantAction, ChatRequest, DataPayload, UserProfile
from .mock_platform import MockPlatform
from .cache import get_caching_service

logger = logging.getLogger(__name__)


class DataProduct(BaseModel):
    """Normalized product representation returned to the assistant layer."""

    id: str
    name: str
    price: float | None = None
    currency: str = "RUB"
    image_url: str | None = None
    availability: str = "in_stock"
    stock_qty: int | None = None
    availability_text: str | None = None
    description: str | None = None
    is_rx: bool = False
    is_for_children: bool = False
    is_sugar_free: bool = False
    is_lactose_free: bool = False
    tags: List[str] = Field(default_factory=list)
    symptom_tags: List[str] = Field(default_factory=list)
    disease_tags: List[str] = Field(default_factory=list)
    category: str | None = None
    pharmacy_id: str | None = None
    region_id: str | None = None
    promo_flags: List[str] = Field(default_factory=list)
    score: float | None = None
    dosage_form: str | None = None

    @property
    def is_available(self) -> bool:
        """True when the SKU can be sold right now."""

        if self.availability not in {"in_stock", "limited"}:
            return False
        if self.stock_qty is not None and self.stock_qty <= 0:
            return False
        return True


class PlatformApiClientError(RuntimeError):
    """Raised when the platform API fails."""


class PlatformApiClient:
    """HTTP client responsible for calling internal platform services."""

    _META_FILTER_ATTRS: Dict[str, str] = {
        "без сахара": "is_sugar_free",
        "sugar_free": "is_sugar_free",
        "для детей": "is_for_children",
        "kids": "is_for_children",
        "без лактозы": "is_lactose_free",
        "lactose_free": "is_lactose_free",
    }

    _PRODUCT_INTENTS = {
        IntentType.FIND_BY_SYMPTOM,
        IntentType.SYMPTOM_TO_PRODUCT,
        IntentType.FIND_BY_DISEASE,
        IntentType.DISEASE_TO_PRODUCT,
        IntentType.FIND_BY_META_FILTERS,
        IntentType.FIND_BY_CATEGORY,
        IntentType.FIND_PRODUCT_BY_NAME,
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._mock = MockPlatform()
        self._cache = get_caching_service()

    def get_user_profile(self, user_id: str) -> Dict[str, Any] | None:
        return self._mock.get_user_profile(user_id)

    def get_user_addresses(self, user_id: str) -> List[Dict[str, Any]]:
        return self._mock.get_user_addresses(user_id)

    def get_pharmacy(self, pharmacy_id: str) -> Dict[str, Any] | None:
        return self._mock.get_pharmacy(pharmacy_id)

    def list_pharmacies(self, parameters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """
        TODO: Replace with geo-aware pharmacy discovery service.
        """

        limit = 5
        if parameters:
            try:
                limit = int(parameters.get("limit", limit))
            except (TypeError, ValueError):
                pass
        return self._mock.list_pharmacies(limit=limit)

    def get_favorites(self, user_id: str) -> List[Dict[str, Any]]:
        favorites = self._mock.get_favorites(user_id)
        return self._serialize_products(favorites)

    def add_favorite(self, user_id: str, product_id: str) -> List[Dict[str, Any]]:
        favorites = self._mock.add_favorite(user_id, product_id)
        return self._serialize_products(favorites)

    def remove_favorite(self, user_id: str, product_id: str) -> List[Dict[str, Any]]:
        favorites = self._mock.remove_favorite(user_id, product_id)
        return self._serialize_products(favorites)

    def get_orders(self, user_id: str, status: str | None = None) -> List[Dict[str, Any]]:
        orders = self._mock.get_orders(user_id, status=status)
        return [self._serialize_order(order) for order in orders]

    def get_recent_purchases(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        return self._mock.get_recent_purchases(user_id, limit=limit)

    def get_recommendations(
        self,
        user_id: str | None,
        context: Dict[str, Any] | None = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        TODO: Integrate with recommendation service.
        """

        if user_id:
            recent = self._mock.get_recent_purchases(user_id, limit=limit)
            if recent:
                return recent
        catalog = self._mock.list_products()
        return [product for product in catalog[:limit]]

    def create_support_ticket(
        self,
        *,
        conversation_id: str | None,
        user_id: str | None,
        issue: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        TODO: Send the ticket to the real customer-support system.
        """

        ticket_id = f"SUP-{uuid.uuid4().hex[:8]}"
        logger.info(
            "Support ticket created placeholder ticket_id=%s user_id=%s conversation_id=%s issue=%s",
            ticket_id,
            user_id,
            conversation_id,
            issue,
        )
        return {
            "ticket_id": ticket_id,
            "status": "queued",
            "issue": issue,
            "payload": payload,
        }

    async def dispatch(
        self,
        action: AssistantAction,
        request: ChatRequest,
        *,
        user_profile: UserProfile | None = None,
    ) -> DataPayload:
        """Route action to platform-specific handler."""

        intent = action.intent
        if not intent:
            logger.info("Action %s does not specify an intent; skipping.", action.type)
            return DataPayload()

        if intent in self._PRODUCT_INTENTS:
            products = await self.fetch_products(
                intent,
                action.parameters,
                request,
                user_profile=user_profile,
            )
            return DataPayload(products=products)

        # Product info handlers
        product_info_intents = {
            IntentType.SHOW_PRODUCT_INFO,
            IntentType.SHOW_PRODUCT_INSTRUCTIONS,
            IntentType.SHOW_PRODUCT_CONTRAINDICATIONS,
            IntentType.SHOW_DETAILED_PRODUCT_SPECIFICATIONS,
        }
        if intent in product_info_intents:
            return await self._handle_product_info(intent, action.parameters, request)

        # Product search handlers
        search_intents = {
            IntentType.FIND_PRODUCT_BY_INN: self.find_by_inn,
            IntentType.FIND_ANALOGS: self.find_analogs,
            IntentType.FIND_PROMO: self.find_promo,
        }
        if intent in search_intents:
            handler = search_intents[intent]
            products = await handler(action.parameters, request, user_profile)
            return DataPayload(products=[p.model_dump() for p in products])

        # Pharmacy handlers
        pharmacy_intents = {
            IntentType.SHOW_PHARMACIES_BY_METRO,
            IntentType.SHOW_PHARMACY_INFO,
            IntentType.SHOW_PHARMACY_HOURS,
            IntentType.SHOW_PRODUCT_AVAILABILITY,
            IntentType.SHOW_NEAREST_PHARMACY_WITH_PRODUCT,
        }
        if intent in pharmacy_intents:
            return await self._handle_pharmacy(intent, action.parameters, request, user_profile)

        # Order handlers
        order_intents = {
            IntentType.PLACE_ORDER,
            IntentType.CANCEL_ORDER,
            IntentType.EXTEND_ORDER,
            IntentType.REORDER_PREVIOUS,
            IntentType.SHOW_ORDER_DELIVERY_TIME,
        }
        if intent in order_intents:
            return await self._handle_order(intent, action.parameters, request)

        # Cart handlers
        cart_handlers: Dict[IntentType, Callable[[Dict[str, Any], ChatRequest], Awaitable[Dict[str, Any]]]] = {
            IntentType.SHOW_CART: self.show_cart,
            IntentType.ADD_TO_CART: self.add_to_cart,
            IntentType.APPLY_PROMO_CODE: self.apply_promo_code,
            IntentType.SELECT_DELIVERY_TYPE: self.select_delivery_type,
        }
        if intent in cart_handlers:
            handler = cart_handlers[intent]
            payload = await handler(action.parameters, request)
            return DataPayload(cart=payload)

        # User handlers
        user_intents = {
            IntentType.UPDATE_PROFILE,
            IntentType.SHOW_ACTIVE_COUPONS,
        }
        if intent in user_intents:
            return await self._handle_user(intent, action.parameters, request)

        # Booking handler
        if intent == IntentType.BOOK_PRODUCT_PICKUP:
            return await self.book_product_pickup(action.parameters, request)

        # Legacy handlers
        handlers: Dict[IntentType, Callable[[Dict[str, Any], ChatRequest], Awaitable[Dict[str, Any]]]] = {
            IntentType.SHOW_ORDER_STATUS: self.show_order_status,
        }

        handler = handlers.get(intent)
        if handler is None:
            logger.info("No platform handler defined for intent %s yet.", intent)
            return DataPayload()

        payload = await handler(action.parameters, request)
        if intent == IntentType.SHOW_ORDER_STATUS:
            return DataPayload(orders=[payload])
        return DataPayload()

    async def fetch_products(
        self,
        intent: IntentType,
        parameters: Dict[str, Any],
        request: ChatRequest,
        *,
        user_profile: UserProfile | None = None,
    ) -> List[Dict[str, Any]]:
        """Return serialized products for the given intent."""

        cache_payload = self._search_slots_payload(intent, parameters, request)
        cached = self._cache.get_product_results(cache_payload)
        if cached:
            return cached

        handlers: Dict[
            IntentType,
            Callable[[Dict[str, Any], ChatRequest, UserProfile | None], Awaitable[List[DataProduct]]],
        ] = {
            IntentType.FIND_BY_SYMPTOM: self.find_by_symptom,
            IntentType.SYMPTOM_TO_PRODUCT: self.find_by_symptom,
            IntentType.FIND_BY_DISEASE: self.find_by_disease,
            IntentType.DISEASE_TO_PRODUCT: self.find_by_disease,
            IntentType.FIND_BY_META_FILTERS: self.find_by_meta_filters,
            IntentType.FIND_BY_CATEGORY: self.find_by_category,
            IntentType.FIND_PRODUCT_BY_NAME: self.find_by_name,
        }

        handler = handlers.get(intent)
        if handler is None:
            logger.info("No product handler registered for intent %s", intent)
            return []

        products = await handler(parameters, request, user_profile)
        serialized = [product.model_dump() for product in products]
        self._cache.set_product_results(cache_payload, serialized, ttl_seconds=600)
        return serialized

    async def find_by_symptom(
        self,
        parameters: Dict[str, Any],
        request: ChatRequest,
        user_profile: UserProfile | None,
    ) -> List[DataProduct]:
        """
        Simulate `GET /search/symptom`.

        TODO: Replace with real search microservice call once the platform endpoint is ready.
        """

        symptom = str(parameters.get("symptom") or parameters.get("query") or "").strip().lower()
        candidates = self._mock_catalog()
        if symptom:
            candidates = [
                product
                for product in candidates
                if self._matches_tag(product.symptom_tags, symptom) or self._matches_tag(product.tags, symptom)
            ]
        meta_filters = parameters.get("meta_filters") or []
        price_limit = self._resolve_price_max(parameters, user_profile)
        return self._post_process_products(
            candidates,
            request,
            include_rx=bool(parameters.get("allow_rx")),
            meta_filters=meta_filters,
            user_profile=user_profile,
            price_max=price_limit,
            intent=IntentType.FIND_BY_SYMPTOM,
        )

    async def find_by_disease(
        self,
        parameters: Dict[str, Any],
        request: ChatRequest,
        user_profile: UserProfile | None,
    ) -> List[DataProduct]:
        """
        Simulate `GET /search/disease`.

        TODO: Plug into the disease-to-product recommender API.
        """

        disease = str(parameters.get("disease") or parameters.get("diagnosis") or "").strip().lower()
        candidates = self._mock_catalog()
        if disease:
            candidates = [
                product
                for product in candidates
                if self._matches_tag(product.disease_tags, disease) or self._matches_tag(product.tags, disease)
            ]
        meta_filters = parameters.get("meta_filters") or []
        price_limit = self._resolve_price_max(parameters, user_profile)
        return self._post_process_products(
            candidates,
            request,
            include_rx=bool(parameters.get("allow_rx")),
            meta_filters=meta_filters,
            user_profile=user_profile,
            price_max=price_limit,
            intent=IntentType.FIND_BY_DISEASE,
        )

    async def find_by_meta_filters(
        self,
        parameters: Dict[str, Any],
        request: ChatRequest,
        user_profile: UserProfile | None,
    ) -> List[DataProduct]:
        """
        Simulate `GET /search/meta`.

        TODO: Call the catalog facet endpoint with attribute filters.
        """

        meta_filters = parameters.get("meta_filters") or parameters.get("filters") or []
        candidates = self._mock_catalog()
        price_limit = self._resolve_price_max(parameters, user_profile)
        return self._post_process_products(
            candidates,
            request,
            include_rx=bool(parameters.get("allow_rx")),
            meta_filters=meta_filters,
            user_profile=user_profile,
            price_max=price_limit,
            intent=IntentType.FIND_BY_META_FILTERS,
        )

    async def find_by_category(
        self,
        parameters: Dict[str, Any],
        request: ChatRequest,
        user_profile: UserProfile | None,
    ) -> List[DataProduct]:
        """
        Simulate `GET /search/category`.

        TODO: Replace with real category aggregation query.
        """

        category = (
            str(parameters.get("category") or parameters.get("category_id") or parameters.get("category_slug") or "")
            .strip()
            .lower()
        )
        candidates = self._mock_catalog()
        if category:
            candidates = [
                product
                for product in candidates
                if product.category and category in product.category.lower()
            ]
        price_limit = self._resolve_price_max(parameters, user_profile)
        return self._post_process_products(
            candidates,
            request,
            include_rx=bool(parameters.get("allow_rx")),
            meta_filters=parameters.get("meta_filters"),
            user_profile=user_profile,
            price_max=price_limit,
            intent=IntentType.FIND_BY_CATEGORY,
        )

    async def find_by_name(
        self,
        parameters: Dict[str, Any],
        request: ChatRequest,
        user_profile: UserProfile | None,
    ) -> List[DataProduct]:
        """
        Simulate `GET /search/by-name`.

        TODO: Integrate with catalog search by product name or barcode.
        """

        query = str(parameters.get("name") or parameters.get("product_name") or request.message or "").strip().lower()
        candidates = self._mock_catalog()
        if query:
            candidates = [
                product
                for product in candidates
                if query in product.name.lower() or self._matches_tag(product.tags, query)
            ]
        price_limit = self._resolve_price_max(parameters, user_profile)
        return self._post_process_products(
            candidates,
            request,
            include_rx=bool(parameters.get("allow_rx")),
            meta_filters=parameters.get("meta_filters"),
            user_profile=user_profile,
            price_max=price_limit,
            intent=IntentType.FIND_PRODUCT_BY_NAME,
        )

    async def show_cart(self, parameters: Dict[str, Any], request: ChatRequest) -> Dict[str, Any]:
        """
        Simulate `GET /cart/{cartId}`.

        TODO: integrate with the cart service using authenticated user tokens.
        """

        cart = self._mock.get_cart(request.user_id)
        selected_pharmacy = request.ui_state.selected_pharmacy_id if request.ui_state else None
        if selected_pharmacy:
            cart["pharmacy_id"] = selected_pharmacy
        return self._serialize_cart(cart)

    async def add_to_cart(self, parameters: Dict[str, Any], request: ChatRequest) -> Dict[str, Any]:
        """
        Simulate `POST /cart/items`.

        TODO: replace with a real write call to the cart service.
        """

        product_id = parameters.get("product_id")
        if not product_id:
            return await self.show_cart(parameters, request)

        qty = int(parameters.get("qty") or parameters.get("quantity") or 1)
        qty = max(qty, 1)
        cart = self._mock.add_to_cart(request.user_id, product_id, qty)
        selected_pharmacy = request.ui_state.selected_pharmacy_id if request.ui_state else None
        if selected_pharmacy:
            cart["pharmacy_id"] = selected_pharmacy
        return self._serialize_cart(cart)

    async def show_order_status(self, parameters: Dict[str, Any], request: ChatRequest) -> Dict[str, Any]:
        """
        Simulate `GET /orders/{orderId}`.

        TODO: connect to orders service for live status tracking.
        """

        order_id = str(parameters.get("order_id") or "").strip()
        order: Dict[str, Any] | None = None
        if order_id:
            order = self._mock.get_order_by_id(order_id)
        if not order and request.user_id:
            active_orders = self._mock.get_orders(request.user_id, status="active")
            if active_orders:
                order = active_orders[0]
        if not order and request.user_id:
            history = self._mock.get_orders(request.user_id)
            if history:
                order = history[0]
        if order:
            selected_pharmacy = request.ui_state.selected_pharmacy_id if request.ui_state else None
            if selected_pharmacy and not order.get("pharmacy_id"):
                order["pharmacy_id"] = selected_pharmacy
            return self._serialize_order(order)
        fallback_id = order_id or "order-demo"
        return self._serialize_order(
            {
            "order_id": fallback_id,
            "status": "ready_for_pickup",
            "status_label": "Готов к выдаче",
            "pharmacy_id": request.ui_state.selected_pharmacy_id if request.ui_state else None,
            "eta_minutes": 25,
            "items_ready": True,
            "items": [],
            }
        )

    # -------------------------------------------------------------------------
    # Product Info Handlers
    # -------------------------------------------------------------------------
    async def _handle_product_info(
        self, intent: IntentType, parameters: Dict[str, Any], request: ChatRequest
    ) -> DataPayload:
        """Handle product information requests."""
        product_id = parameters.get("product_id")
        if not product_id:
            return DataPayload(message="Пожалуйста, укажите товар.")

        product = self._get_product_by_id(product_id)
        if not product:
            return DataPayload(message="Товар не найден.")

        if intent == IntentType.SHOW_PRODUCT_INFO:
            return DataPayload(products=[product.model_dump()])

        if intent == IntentType.SHOW_PRODUCT_INSTRUCTIONS:
            instructions = self._mock.get_product_instructions(product_id)
            return DataPayload(
                products=[product.model_dump()],
                message=instructions or "Инструкция не найдена. Обратитесь к упаковке или фармацевту.",
            )

        if intent == IntentType.SHOW_PRODUCT_CONTRAINDICATIONS:
            contraindications = self._mock.get_product_contraindications(product_id)
            return DataPayload(
                products=[product.model_dump()],
                message=contraindications or "Информация о противопоказаниях не найдена.",
            )

        if intent == IntentType.SHOW_DETAILED_PRODUCT_SPECIFICATIONS:
            specs = self._mock.get_product_specifications(product_id)
            return DataPayload(
                products=[product.model_dump()],
                metadata={"specifications": specs} if specs else {},
            )

        return DataPayload(products=[product.model_dump()])

    # -------------------------------------------------------------------------
    # Search Handlers
    # -------------------------------------------------------------------------
    async def find_by_inn(
        self, parameters: Dict[str, Any], request: ChatRequest, user_profile: UserProfile | None
    ) -> List[DataProduct]:
        """Find products by INN (International Nonproprietary Name)."""
        inn = str(parameters.get("inn") or parameters.get("query") or "").strip()
        if not inn:
            return []

        raw_products = self._mock.find_products_by_inn(inn)
        products = []
        for raw in raw_products:
            try:
                products.append(DataProduct.model_validate(raw))
            except ValidationError:
                continue

        price_limit = self._resolve_price_max(parameters, user_profile)
        return self._post_process_products(
            products, request, include_rx=False, user_profile=user_profile, price_max=price_limit
        )

    async def find_analogs(
        self, parameters: Dict[str, Any], request: ChatRequest, user_profile: UserProfile | None
    ) -> List[DataProduct]:
        """Find analog products."""
        product_id = parameters.get("product_id")
        if not product_id:
            return []

        raw_products = self._mock.find_analogs(product_id)
        products = []
        for raw in raw_products:
            try:
                products.append(DataProduct.model_validate(raw))
            except ValidationError:
                continue

        price_limit = self._resolve_price_max(parameters, user_profile)
        return self._post_process_products(
            products, request, include_rx=False, user_profile=user_profile, price_max=price_limit
        )

    async def find_promo(
        self, parameters: Dict[str, Any], request: ChatRequest, user_profile: UserProfile | None
    ) -> List[DataProduct]:
        """Find products with promotions."""
        limit = int(parameters.get("limit", 10))
        raw_products = self._mock.find_promo_products(limit=limit)
        products = []
        for raw in raw_products:
            try:
                products.append(DataProduct.model_validate(raw))
            except ValidationError:
                continue

        price_limit = self._resolve_price_max(parameters, user_profile)
        return self._post_process_products(
            products, request, include_rx=False, user_profile=user_profile, price_max=price_limit
        )

    # -------------------------------------------------------------------------
    # Pharmacy Handlers
    # -------------------------------------------------------------------------
    async def _handle_pharmacy(
        self,
        intent: IntentType,
        parameters: Dict[str, Any],
        request: ChatRequest,
        user_profile: UserProfile | None,
    ) -> DataPayload:
        """Handle pharmacy-related requests."""

        if intent == IntentType.SHOW_PHARMACIES_BY_METRO:
            metro = str(parameters.get("metro") or parameters.get("metro_station") or "").strip()
            pharmacies = self._mock.find_pharmacies_by_metro(metro) if metro else self._mock.list_pharmacies(limit=5)
            return DataPayload(pharmacies=pharmacies)

        if intent == IntentType.SHOW_PHARMACY_INFO:
            pharmacy_id = parameters.get("pharmacy_id")
            if not pharmacy_id:
                return DataPayload(message="Пожалуйста, укажите аптеку.")
            pharmacy = self._mock.get_pharmacy(pharmacy_id)
            if not pharmacy:
                return DataPayload(message="Аптека не найдена.")
            return DataPayload(pharmacies=[pharmacy])

        if intent == IntentType.SHOW_PHARMACY_HOURS:
            pharmacy_id = parameters.get("pharmacy_id")
            if not pharmacy_id:
                return DataPayload(message="Пожалуйста, укажите аптеку.")
            hours = self._mock.get_pharmacy_working_hours(pharmacy_id)
            pharmacy = self._mock.get_pharmacy(pharmacy_id)
            return DataPayload(
                pharmacies=[pharmacy] if pharmacy else [],
                message=f"Режим работы: {hours}" if hours else "Информация о режиме работы не найдена.",
            )

        if intent == IntentType.SHOW_PRODUCT_AVAILABILITY:
            product_id = parameters.get("product_id")
            if not product_id:
                return DataPayload(message="Пожалуйста, укажите товар.")
            availability = self._mock.get_product_availability_by_pharmacies(product_id)
            product = self._get_product_by_id(product_id)
            return DataPayload(
                products=[product.model_dump()] if product else [],
                pharmacies=availability,
            )

        if intent == IntentType.SHOW_NEAREST_PHARMACY_WITH_PRODUCT:
            product_id = parameters.get("product_id")
            if not product_id:
                return DataPayload(message="Пожалуйста, укажите товар.")
            region = None
            if user_profile and user_profile.preferences:
                region = user_profile.preferences.region
            result = self._mock.find_nearest_pharmacy_with_product(product_id, region)
            if not result:
                return DataPayload(message="К сожалению, товар не найден в ближайших аптеках.")
            return DataPayload(
                products=[result["product"]],
                pharmacies=[result["pharmacy"]],
            )

        return DataPayload()

    # -------------------------------------------------------------------------
    # Order Handlers
    # -------------------------------------------------------------------------
    async def _handle_order(
        self, intent: IntentType, parameters: Dict[str, Any], request: ChatRequest
    ) -> DataPayload:
        """Handle order-related requests."""
        user_id = request.user_id
        if not user_id:
            return DataPayload(message="Для работы с заказами необходима авторизация.")

        if intent == IntentType.PLACE_ORDER:
            cart = self._mock.get_cart(user_id)
            items = cart.get("items", [])
            if not items:
                return DataPayload(message="Корзина пуста. Добавьте товары для оформления заказа.")
            delivery_type = parameters.get("delivery_type", "pickup")
            order = self._mock.create_order(user_id, items, delivery_type)
            self._mock.clear_cart(user_id)
            return DataPayload(orders=[self._serialize_order(order)])

        if intent == IntentType.CANCEL_ORDER:
            order_id = parameters.get("order_id")
            if not order_id:
                return DataPayload(message="Пожалуйста, укажите номер заказа.")
            reason = parameters.get("reason")
            order = self._mock.cancel_order(order_id, reason)
            if not order:
                return DataPayload(message="Не удалось отменить заказ. Возможно, он уже выполнен.")
            return DataPayload(orders=[self._serialize_order(order)])

        if intent == IntentType.EXTEND_ORDER:
            order_id = parameters.get("order_id")
            if not order_id:
                return DataPayload(message="Пожалуйста, укажите номер заказа.")
            order = self._mock.extend_order_pickup_time(order_id)
            if not order:
                return DataPayload(message="Не удалось продлить время хранения заказа.")
            return DataPayload(
                orders=[self._serialize_order(order)],
                message="Время хранения заказа продлено до конца дня.",
            )

        if intent == IntentType.REORDER_PREVIOUS:
            order_id = parameters.get("order_id")
            if not order_id:
                orders = self._mock.get_orders(user_id, status="completed")
                if orders:
                    order_id = orders[0].get("order_id")
            if not order_id:
                return DataPayload(message="Не найден предыдущий заказ для повтора.")
            new_order = self._mock.reorder(user_id, order_id)
            if not new_order:
                return DataPayload(message="Не удалось повторить заказ.")
            return DataPayload(orders=[self._serialize_order(new_order)])

        if intent == IntentType.SHOW_ORDER_DELIVERY_TIME:
            order_id = parameters.get("order_id")
            if order_id:
                order = self._mock.get_order_by_id(order_id)
            else:
                orders = self._mock.get_orders(user_id, status="active")
                order = orders[0] if orders else None
            if not order:
                return DataPayload(message="Активный заказ не найден.")
            return DataPayload(
                orders=[self._serialize_order(order)],
                message=f"Ожидаемое время: {order.get('estimated_delivery', 'уточняется')}",
            )

        return DataPayload()

    # -------------------------------------------------------------------------
    # Cart Extended Handlers
    # -------------------------------------------------------------------------
    async def apply_promo_code(self, parameters: Dict[str, Any], request: ChatRequest) -> Dict[str, Any]:
        """Apply promo code to cart."""
        promo_code = str(parameters.get("promo_code") or parameters.get("code") or "").strip()
        if not promo_code:
            cart = self._mock.get_cart(request.user_id)
            return {**self._serialize_cart(cart), "promo_error": "Пожалуйста, укажите промокод."}

        cart = self._mock.apply_promo_code(request.user_id, promo_code)
        return self._serialize_cart(cart)

    async def select_delivery_type(self, parameters: Dict[str, Any], request: ChatRequest) -> Dict[str, Any]:
        """Select delivery type for order."""
        delivery_type = parameters.get("delivery_type", "pickup")
        cart = self._mock.get_cart(request.user_id)
        cart["delivery_type"] = delivery_type
        cart["delivery_info"] = (
            "Доставка курьером 1-2 рабочих дня" if delivery_type == "delivery" else "Самовывоз из аптеки"
        )
        return self._serialize_cart(cart)

    # -------------------------------------------------------------------------
    # User Handlers
    # -------------------------------------------------------------------------
    async def _handle_user(
        self, intent: IntentType, parameters: Dict[str, Any], request: ChatRequest
    ) -> DataPayload:
        """Handle user-related requests."""
        user_id = request.user_id
        if not user_id:
            return DataPayload(message="Для этого действия необходима авторизация.")

        if intent == IntentType.UPDATE_PROFILE:
            preferences = parameters.get("preferences", {})
            if not preferences:
                return DataPayload(message="Укажите, что именно нужно обновить в профиле.")
            user = self._mock.update_user_preferences(user_id, preferences)
            if not user:
                return DataPayload(message="Не удалось обновить профиль.")
            return DataPayload(
                metadata={"user": user},
                message="Профиль успешно обновлён.",
            )

        if intent == IntentType.SHOW_ACTIVE_COUPONS:
            coupons = self._mock.get_user_coupons(user_id)
            return DataPayload(
                metadata={"coupons": coupons},
                message=f"У вас {len(coupons)} активных купонов." if coupons else "Активных купонов нет.",
            )

        return DataPayload()

    # -------------------------------------------------------------------------
    # Booking Handler
    # -------------------------------------------------------------------------
    async def book_product_pickup(
        self, parameters: Dict[str, Any], request: ChatRequest
    ) -> DataPayload:
        """Book a product for pickup."""
        user_id = request.user_id
        if not user_id:
            return DataPayload(message="Для бронирования необходима авторизация.")

        product_id = parameters.get("product_id")
        pharmacy_id = parameters.get("pharmacy_id")
        if not product_id:
            return DataPayload(message="Пожалуйста, укажите товар для бронирования.")
        if not pharmacy_id:
            ui_state = request.ui_state
            pharmacy_id = ui_state.selected_pharmacy_id if ui_state else None
        if not pharmacy_id:
            return DataPayload(message="Пожалуйста, выберите аптеку для бронирования.")

        booking = self._mock.book_product_pickup(user_id, product_id, pharmacy_id)
        if not booking:
            return DataPayload(message="К сожалению, не удалось забронировать товар. Возможно, он отсутствует в наличии.")

        return DataPayload(
            products=[booking["product"]],
            pharmacies=[booking["pharmacy"]],
            metadata={"booking": {"id": booking["booking_id"], "status": booking["status"]}},
            message=booking["message"],
        )

    def _search_slots_payload(
        self, intent: IntentType, parameters: Dict[str, Any], request: ChatRequest
    ) -> Dict[str, Any]:
        tracked_keys = {"symptom", "age", "price_max", "dosage_form", "name"}
        payload = {key: parameters.get(key) for key in tracked_keys if key in parameters}
        payload["intent"] = intent.value
        ui_state = request.ui_state
        if ui_state:
            payload["region"] = ui_state.selected_region_id
            payload["pharmacy"] = ui_state.selected_pharmacy_id
        return payload

    def _resolve_price_max(self, parameters: Dict[str, Any], user_profile: UserProfile | None) -> float | None:
        """Resolve price_max from parameters or user profile with validation."""
        upper_bound = self._settings.price_max_upper_bound
        
        raw = parameters.get("price_max")
        if raw is not None:
            try:
                price = float(raw)
                if price < 0:
                    logger.warning("Negative price_max=%s ignored", price)
                    return None
                if price > upper_bound:
                    logger.warning("price_max=%s exceeds upper bound %s, clamping", price, upper_bound)
                    return upper_bound
                return price
            except (TypeError, ValueError):
                logger.debug("Invalid price_max value %s", raw)
        
        if user_profile:
            prefs = user_profile.preferences
            profile_price = getattr(prefs, "default_max_price", None) or getattr(prefs, "price_ceiling", None)
            if profile_price is not None:
                try:
                    price = float(profile_price)
                    if 0 < price <= upper_bound:
                        return price
                    if price > upper_bound:
                        return upper_bound
                except (TypeError, ValueError):
                    pass
        return None

    def _post_process_products(
        self,
        products: List[DataProduct],
        request: ChatRequest,
        *,
        include_rx: bool = False,
        meta_filters: Sequence[str] | None = None,
        user_profile: UserProfile | None = None,
        price_max: float | None = None,
        intent: IntentType | None = None,
    ) -> List[DataProduct]:
        filtered = self._filter_products(
            products,
            request,
            include_rx=include_rx,
            meta_filters=meta_filters,
            user_profile=user_profile,
            price_max=price_max,
        )
        return self._sort_products(filtered, user_profile=user_profile, intent=intent)

    def _filter_products(
        self,
        products: List[DataProduct],
        request: ChatRequest,
        *,
        include_rx: bool,
        meta_filters: Sequence[str] | None,
        user_profile: UserProfile | None,
        price_max: float | None,
    ) -> List[DataProduct]:
        ui_state = request.ui_state
        region_id = ui_state.selected_region_id if ui_state else None
        pharmacy_id = ui_state.selected_pharmacy_id if ui_state else None
        normalized_filters = self._normalize_meta_filters(meta_filters)
        preferences = user_profile.preferences if user_profile else None
        sugar_free_pref = preferences.sugar_free is True if preferences else False
        lactose_free_pref = preferences.lactose_free is True if preferences else False
        for_children_pref = preferences.for_children is True if preferences else False
        preferred_forms = self._normalize_preferred_forms(preferences)
        effective_region = region_id or (preferences.region if preferences else None)
        # TODO: replace client-side region filtering with real geo-aware catalog queries.

        filtered: List[DataProduct] = []
        general_passed: List[DataProduct] = []
        preference_filters_applied = False
        for product in products:
            if effective_region and product.region_id and product.region_id != effective_region:
                continue
            if pharmacy_id and product.pharmacy_id and product.pharmacy_id != pharmacy_id:
                continue
            if not include_rx and product.is_rx:
                continue
            if not product.is_available:
                continue
            if not self._match_meta_filters(product, normalized_filters):
                continue
            if price_max is not None and product.price is not None and product.price > price_max:
                continue
            general_passed.append(product)
            if preferences:
                if sugar_free_pref:
                    preference_filters_applied = True
                    if not product.is_sugar_free:
                        continue
                if lactose_free_pref:
                    preference_filters_applied = True
                    if not product.is_lactose_free:
                        continue
                if for_children_pref:
                    preference_filters_applied = True
                    if not product.is_for_children:
                        continue
                if preferred_forms:
                    preference_filters_applied = True
                    if not self._preferred_form_matches(product, preferred_forms):
                        continue
            filtered.append(product)
        if not filtered and preference_filters_applied:
            # Fall back to general filters if strict preference filters produced no results.
            return general_passed
        return filtered

    def _sort_products(
        self,
        products: List[DataProduct],
        *,
        user_profile: UserProfile | None,
        intent: IntentType | None,
    ) -> List[DataProduct]:
        # При has_children приоритизируем детские товары в начале списка
        has_children = False
        if user_profile:
            has_children = getattr(user_profile.preferences, "has_children", False)
        
        def sort_key(product: DataProduct) -> tuple:
            # Детские товары идут первыми если у пользователя есть дети
            children_priority = 0 if (has_children and product.is_for_children) else 1
            return (
                children_priority,
                self._preference_priority(product, user_profile, intent),
                0 if product.is_available else 1,
                0 if product.promo_flags else 1,
                -(product.stock_qty or 0),
                product.price if product.price is not None else float("inf"),
            )
        
        return sorted(products, key=sort_key)

    def _normalize_meta_filters(self, meta_filters: Sequence[str] | None) -> List[str]:
        if not meta_filters:
            return []
        normalized: List[str] = []
        for value in meta_filters:
            if not isinstance(value, str):
                continue
            normalized.append(value.strip().lower())
        return normalized

    def _match_meta_filters(self, product: DataProduct, meta_filters: Sequence[str]) -> bool:
        for value in meta_filters:
            attr = self._META_FILTER_ATTRS.get(value)
            if attr and not getattr(product, attr, False):
                return False
        return True

    def _matches_tag(self, tags: Sequence[str], needle: str) -> bool:
        needle = needle.lower()
        return any(needle in tag.lower() for tag in tags)

    def _preference_priority(
        self,
        product: DataProduct,
        user_profile: UserProfile | None,
        intent: IntentType | None,
    ) -> int:
        if not user_profile:
            return 0
        prefs = user_profile.preferences
        score = 0
        if prefs.sugar_free:
            score += -3 if product.is_sugar_free else 3
        if prefs.lactose_free:
            score += -2 if product.is_lactose_free else 2
        if prefs.for_children:
            score += -3 if product.is_for_children else 2
        preferred_forms = getattr(prefs, "preferred_dosage_forms", None) or getattr(prefs, "preferred_forms", None) or []
        if preferred_forms:
            if self._matches_preferred_form(product, preferred_forms):
                score -= 1
            else:
                score += 1
        # Если у пользователя есть дети, предпочитаем детские версии препаратов
        # для всех product-related intents
        if getattr(prefs, "has_children", False) and intent in self._PRODUCT_INTENTS:
            score += -2 if product.is_for_children else 1
        tags = user_profile.tags or []
        if "promo_lover" in tags:
            score -= 1 if product.promo_flags else 0
        return score

    def _normalize_preferred_forms(self, preferences) -> List[str]:
        if not preferences:
            return []
        forms = (
            getattr(preferences, "preferred_dosage_forms", None)
            or getattr(preferences, "preferred_forms", None)
            or []
        )
        normalized: List[str] = []
        for form in forms:
            if not form:
                continue
            normalized.append(str(form).strip().lower())
        return normalized

    def _preferred_form_matches(self, product: DataProduct, normalized_forms: Sequence[str]) -> bool:
        if not normalized_forms:
            return True
        form_hint = self._product_form_hint(product)
        if form_hint and form_hint in normalized_forms:
            return True
        return self._matches_preferred_form(product, normalized_forms)

    def _product_form_hint(self, product: DataProduct) -> str | None:
        if product.dosage_form:
            return str(product.dosage_form).strip().lower()
        haystack = (product.name or "").lower()
        description = (product.description or "").lower()
        combined = f"{haystack} {description}".strip()
        form_map = {
            "tablets": ["таблет", "табл"],
            "syrup": ["сироп"],
            "spray": ["спрей"],
            "capsules": ["капсул"],
            "drops": ["капл"],
            "powder": ["порошок"],
        }
        for form, keywords in form_map.items():
            if any(keyword in combined for keyword in keywords):
                return form
        return None

    def _matches_preferred_form(self, product: DataProduct, preferred_forms: Sequence[str]) -> bool:
        if not preferred_forms:
            return False
        haystacks = [product.name.lower()]
        if product.description:
            haystacks.append(product.description.lower())
        haystacks.extend(tag.lower() for tag in product.tags)
        for form in preferred_forms:
            normalized = form.strip().lower()
            if not normalized:
                continue
            if any(normalized in hay for hay in haystacks):
                return True
        return False

    def _mock_catalog(self) -> List[DataProduct]:
        """Load product catalog from the mock JSON store."""

        products: List[DataProduct] = []
        for raw in self._mock.list_products():
            try:
                products.append(DataProduct.model_validate(raw))
            except ValidationError as exc:
                logger.warning("Skip invalid mock product entry: %s", exc)
        return products

    def _get_product_by_id(self, product_id: str | None) -> DataProduct | None:
        if not product_id:
            return None
        raw_product = self._mock.get_product(product_id)
        if not raw_product:
            return None
        try:
            return DataProduct.model_validate(raw_product)
        except ValidationError as exc:
            logger.warning("Invalid product %s in mock catalog: %s", product_id, exc)
            return None

    def _serialize_products(self, products: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for raw in products:
            try:
                normalized.append(DataProduct.model_validate(raw).model_dump())
            except ValidationError as exc:
                logger.warning("Skip invalid product entry: %s", exc)
        return normalized

    def _serialize_cart(self, cart: Dict[str, Any]) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        items_count = 0
        for item in cart.get("items") or []:
            qty = int(item.get("qty") or 0)
            items_count += qty
            items.append(
                {
                    "product_id": item.get("product_id"),
                    "title": item.get("title"),
                    "qty": qty,
                    "price": self._safe_float(item.get("price")),
                    "image_url": item.get("image_url"),
                }
            )
        snapshot = {
            "cart_id": cart.get("cart_id"),
            "currency": cart.get("currency") or "RUB",
            "items_count": items_count,
            "total_amount": self._safe_float(cart.get("total")),
            "items": items,
            "pharmacy_id": cart.get("pharmacy_id"),
        }
        if cart.get("note"):
            snapshot["note"] = cart["note"]
        if cart.get("subtitle"):
            snapshot["subtitle"] = cart["subtitle"]
        return snapshot

    def _serialize_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        for item in order.get("items") or []:
            items.append(
                {
                    "product_id": item.get("product_id"),
                    "title": item.get("title"),
                    "qty": int(item.get("qty") or 0),
                    "price": self._safe_float(item.get("price")),
                }
            )
        pharmacy_info: Dict[str, Any] | None = None
        pharmacy_id = order.get("pharmacy_id")
        if pharmacy_id:
            pharmacy = self.get_pharmacy(pharmacy_id)
            if pharmacy:
                pharmacy_info = {
                    "id": pharmacy.get("id"),
                    "name": pharmacy.get("name"),
                    "address": pharmacy.get("address"),
                    "lat": pharmacy.get("lat"),
                    "lon": pharmacy.get("lon"),
                }
        payload: Dict[str, Any] = {
            "order_id": order.get("order_id"),
            "number": order.get("number") or order.get("order_id"),
            "status": order.get("status"),
            "status_label": order.get("status_label"),
            "delivery_type": order.get("delivery_type"),
            "pharmacy_id": order.get("pharmacy_id"),
            "created_at": order.get("created_at"),
            "completed_at": order.get("completed_at"),
            "eta_minutes": order.get("eta_minutes"),
            "items_ready": order.get("items_ready"),
            "total_amount": self._safe_float(order.get("total_amount") or order.get("total")),
            "currency": order.get("currency") or "RUB",
            "items": items,
        }
        if order.get("user_id"):
            payload["user_id"] = order["user_id"]
        if pharmacy_info:
            payload["pharmacy"] = pharmacy_info
        return payload

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    async def _get(self, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Reusable GET helper for future real HTTP integrations."""

        url = f"{self._settings.platform_base_url}{path}"
        timeout = httpx.Timeout(self._settings.http_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:
                logger.error("Platform API error calling %s: %s", url, exc)
                raise PlatformApiClientError(str(exc)) from exc

