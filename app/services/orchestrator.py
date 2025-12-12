from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from langsmith import traceable

from ..config import Settings
from ..intents import (
    ActionChannel,
    ActionType,
    IntentType,
    NAVIGATION_INTENTS,
    SYMPTOM_INTENTS,
    get_intent_category,
)
from ..models import (
    AssistantAction,
    AssistantMeta,
    AssistantResponse,
    ChatRequest,
    ChatResponse,
    DataPayload,
    Reply,
    UserProfile,
)
from .assistant_client import AssistantClient
from .debug_meta import DebugMetaBuilder
from .metrics import get_metrics_service
from .dialog_state_store import DialogStateStore, get_dialog_state_store
from .platform_client import PlatformApiClient
from .response_helpers import (
    UNKNOWN_FALLBACK_REPLY,
    LOW_CONFIDENCE_FALLBACK,
    SMALL_TALK_FALLBACK_REPLY,
    PRODUCT_RESULT_LIMIT,
    default_quick_replies,
    symptom_clarification_quick_replies as _symptom_clarification_quick_replies,
    no_results_quick_replies as _no_results_quick_replies,
    merge_quick_replies as _merge_quick_replies,
    has_non_empty_value as _has_non_empty_value,
    missing_symptom_parameters as _missing_symptom_parameters,
    build_symptom_clarification_text as _build_symptom_clarification_text,
    no_results_reply_text as _no_results_reply_text,
    deduplicate_products as _deduplicate_products,
    build_purchase_history_quick_replies,
    most_frequent_item,
    SYMPTOM_PARAM_ALIASES as _SYMPTOM_PARAM_ALIASES,
    SYMPTOM_FIELD_QUESTIONS as _SYMPTOM_FIELD_QUESTIONS,
    SYMPTOM_EXTRA_HINT as _SYMPTOM_EXTRA_HINT,
)
from .safety_filter import SafetyFilter
from .slot_manager import SlotManager
from .router import RouterResult
from .static_responses import get_legal_response, is_legal_intent
from .user_profile_store import UserProfileStore, get_user_profile_store
from ..utils.logging import get_request_logger

logger = logging.getLogger(__name__)

PRODUCT_INTENTS = {
    IntentType.FIND_BY_SYMPTOM,
    IntentType.FIND_BY_DISEASE,
    IntentType.SYMPTOM_TO_PRODUCT,
    IntentType.DISEASE_TO_PRODUCT,
    IntentType.FIND_BY_META_FILTERS,
    IntentType.FIND_BY_CATEGORY,
    IntentType.FIND_PRODUCT_BY_NAME,
}

DATA_DRIVEN_NAV_INTENTS = {
    IntentType.SHOW_CART,
    IntentType.SHOW_PROFILE,
    IntentType.SHOW_FAVORITES,
}

LEGAL_INFO_INTENTS = {
    IntentType.RETURN_POLICY,
    IntentType.STORAGE_RULES,
    IntentType.EXPIRATION_INFO,
    IntentType.PRESCRIPTION_POLICY,
    IntentType.DELIVERY_RULES,
    IntentType.PHARMACY_LEGAL_INFO,
    IntentType.SAFETY_WARNINGS,
    IntentType.SYMPTOM_SELFCARE_ADVICE,
    IntentType.PREVENTION_ADVICE,
    IntentType.ASK_PHARMACIST,
}


class Orchestrator:
    """Maps assistant actions into platform calls and assembles the final payload."""

    def __init__(
        self,
        platform_client: PlatformApiClient,
        assistant_client: AssistantClient,
        settings: Settings,
        user_profile_store: UserProfileStore | None = None,
        dialog_state_store: DialogStateStore | None = None,
    ) -> None:
        self._platform_client = platform_client
        self._assistant_client = assistant_client
        self._settings = settings
        self._user_profile_store = user_profile_store or get_user_profile_store()
        self._dialog_state_store = dialog_state_store or get_dialog_state_store()

    @traceable(run_type="chain", name="orchestrator_build_response")
    async def build_response(
        self,
        *,
        request: ChatRequest,
        assistant_response: AssistantResponse,
        router_matched: bool = False,
        debug_builder: DebugMetaBuilder | None = None,
        trace_id: str | None = None,
    ) -> ChatResponse:
        llm_used, llm_backend = self._infer_llm_context(assistant_response)
        return await self._build_chat_response(
            request=request,
            assistant_response=assistant_response,
            router_matched=router_matched,
            llm_used=llm_used,
            llm_backend=llm_backend,
            debug_builder=debug_builder,
            trace_id=trace_id,
        )

    @traceable(run_type="chain", name="orchestrator_build_response_from_router")
    async def build_response_from_router(
        self,
        *,
        request: ChatRequest,
        router_result: RouterResult,
        slot_manager: SlotManager,
        conversation_id: str,
        user_profile: UserProfile | None,
        debug_builder: DebugMetaBuilder | None = None,
        trace_id: str | None = None,
    ) -> ChatResponse:
        assistant_response = slot_manager.handle_router_result(
            router_result=router_result,
            conversation_id=conversation_id,
            user_profile=user_profile,
            debug_builder=debug_builder,
            trace_id=trace_id,
        )
        metrics = get_metrics_service()
        slot_prompt_pending = False
        if assistant_response.meta and assistant_response.meta.debug:
            slot_prompt_pending = bool(assistant_response.meta.debug.get("slot_prompt_pending"))
        if slot_prompt_pending:
            metrics.record_slot_prompt()
        else:
            metrics.record_slot_success()
        return await self._build_chat_response(
            request=request,
            assistant_response=assistant_response,
            router_matched=True,
            llm_used=False,
            llm_backend="router",
            debug_builder=debug_builder,
            trace_id=trace_id,
        )

    @traceable(run_type="chain", name="orchestrator_build_chat_response")
    async def _build_chat_response(
        self,
        *,
        request: ChatRequest,
        assistant_response: AssistantResponse,
        router_matched: bool,
        llm_used: bool,
        llm_backend: str | None,
        debug_builder: DebugMetaBuilder | None,
        trace_id: str | None,
    ) -> ChatResponse:
        conversation_id = request.conversation_id or ""
        request_logger = get_request_logger(
            logger,
            trace_id=trace_id,
            user_id=request.user_id,
            conversation_id=conversation_id,
        )
        platform_data = DataPayload()
        aggregated_products: List[Dict[str, Any]] = []
        product_query_attempted = False
        additional_notes: list[str] = []
        slot_filling_used = False
        builder = debug_builder or DebugMetaBuilder(request_id=conversation_id or None)
        builder.set_router_matched(router_matched)
        if debug_builder is None:
            builder.set_llm_used(llm_used)

        def _append_orders(new_orders: Sequence[Dict[str, Any]] | None) -> None:
            if not new_orders:
                return
            existing_ids = {
                order.get("order_id")
                for order in platform_data.orders
                if order.get("order_id")
            }
            for order in new_orders:
                order_id = order.get("order_id")
                if order_id and order_id in existing_ids:
                    continue
                platform_data.orders.append(order)
                if order_id:
                    existing_ids.add(order_id)
        meta = assistant_response.meta.model_copy() if assistant_response.meta else AssistantMeta()
        builder.merge_existing(meta.debug)
        confidence = meta.confidence
        threshold = self._settings.assistant_min_confidence
        low_confidence = confidence is not None and confidence < threshold
        user_id = (request.user_id or "").strip()
        actions = list(assistant_response.actions)
        if user_id and actions:
            self._capture_preferences_from_actions(user_id, actions)
        user_profile: UserProfile | None = None
        if user_id:
            user_profile = self._user_profile_store.get_or_create(user_id)

        top_intent = actions[0].intent if actions and actions[0].intent else None
        channel = self._resolve_channel(actions[0]) if actions else None
        if meta and meta.debug:
            slot_filling_used = bool(meta.debug.get("slot_filling_used", False))
            builder.set_slot_filling_used(slot_filling_used)
            if "slot_prompt_pending" in meta.debug:
                builder.set_pending_slots(bool(meta.debug.get("slot_prompt_pending")))
        builder.set_trace_id(trace_id)
        builder.add_intent(getattr(top_intent, "value", None))
        request_logger.info(
            "Building chat response llm_used=%s backend=%s router_matched=%s actions=%s",
            llm_used,
            llm_backend,
            router_matched,
            [
                {
                    "type": getattr(action.type, "value", action.type),
                    "intent": getattr(action.intent, "value", action.intent),
                }
                for action in actions
            ],
        )
        top_category = get_intent_category(top_intent)
        symptom_missing_fields: List[str] = []
        if top_category == "symptom" and actions:
            symptom_missing_fields = _missing_symptom_parameters(actions[0].parameters)
        if router_matched:
            symptom_missing_fields = []
        needs_symptom_clarification = bool(symptom_missing_fields)
        navigation_mode = top_category == "navigation"
        skip_reason: str | None = None
        if top_intent == IntentType.UNKNOWN:
            skip_reason = "unknown_intent"
        elif top_intent == IntentType.SMALL_TALK:
            skip_reason = "small_talk"
        elif low_confidence:
            skip_reason = "low_confidence"

        executed_data_intents: set[IntentType] = set()
        pending_data_intents: set[IntentType] = set()

        def _log_platform_call(method: str, params: Dict[str, Any] | None = None) -> None:
            request_logger.info("Platform call %s params=%s", method, params or {})

        async def _fulfill_data_intent(intent: IntentType, parameters: Dict[str, Any]) -> None:
            if intent == IntentType.SHOW_CART:
                _log_platform_call("show_cart", parameters)
                cart_snapshot = await self._platform_client.show_cart(parameters, request, trace_id=trace_id)
                if cart_snapshot:
                    platform_data.cart = cart_snapshot
                return
            if intent == IntentType.SHOW_PROFILE:
                if not user_id:
                    logger.info("SHOW_PROFILE requested without user_id; skipping.")
                    return
                _log_platform_call("get_user_profile", {"user_id": user_id})
                profile = self._platform_client.get_user_profile(user_id, trace_id=trace_id)
                addresses = self._platform_client.get_user_addresses(user_id, trace_id=trace_id)
                if profile:
                    platform_data.user_profile = profile
                if addresses:
                    platform_data.user_addresses = addresses
                return
            if intent == IntentType.SHOW_FAVORITES:
                if not user_id:
                    logger.info("SHOW_FAVORITES requested without user_id; skipping.")
                    return
                _log_platform_call("get_favorites", {"user_id": user_id})
                platform_data.favorites = self._platform_client.get_favorites(user_id, trace_id=trace_id)

        for action in actions:
            logger.info("Processing action type=%s intent=%s", action.type, action.intent)
            action_channel = self._resolve_channel(action)
            builder.add_intent(getattr(action.intent, "value", action.intent) if action.intent else None)
            request_logger.info(
                "Processing action type=%s intent=%s", getattr(action.type, "value", action.type), action.intent
            )
            if skip_reason and action.type == ActionType.CALL_PLATFORM_API:
                logger.info("Skipping platform call due to %s (intent=%s)", skip_reason, action.intent)
                continue
            if action_channel == ActionChannel.SUPPORT or action.intent == IntentType.CONTACT_SUPPORT:
                ticket = self._handle_support_action(
                    action=action,
                    request=request,
                    user_profile=user_profile,
                    platform_data=platform_data,
                )
                if ticket:
                    additional_notes.append(
                        f"Передал запрос в службу поддержки, номер обращения {ticket.get('ticket_id')}."
                    )
                continue
            
            # Handle legal/policy intents with static responses
            if action.intent and action.intent in LEGAL_INFO_INTENTS:
                legal_response = get_legal_response(action.intent)
                if legal_response:
                    platform_data.message = legal_response
                    logger.info("Handled legal intent %s with static response.", action.intent)
                continue
            if action.type == ActionType.CALL_PLATFORM_API:
                if (
                    needs_symptom_clarification
                    and action.intent
                    and action.intent in SYMPTOM_INTENTS
                ):
                    logger.info(
                        "Delaying symptom intent %s until parameters %s are provided.",
                        action.intent,
                        ", ".join(symptom_missing_fields),
                    )
                    continue
                if (
                    navigation_mode
                    and action.intent
                    and action.intent in NAVIGATION_INTENTS
                    and action.intent not in DATA_DRIVEN_NAV_INTENTS
                ):
                    logger.info(
                        "Navigation intent %s handled via UI hint; skipping platform call.",
                        action.intent,
                    )
                    continue
                intent = action.intent
                if intent and intent in PRODUCT_INTENTS:
                    product_query_attempted = True
                    _log_platform_call(
                        "fetch_products",
                        {
                            "intent": getattr(intent, "value", intent),
                            "region": getattr(request.ui_state, "selected_region_id", None) if request.ui_state else None,
                            "symptom": action.parameters.get("symptom"),
                            "product_name": action.parameters.get("product_name") or action.parameters.get("name"),
                        },
                    )
                    products = await self._platform_client.fetch_products(
                        intent,
                        action.parameters,
                        request,
                        user_profile=user_profile,
                        trace_id=trace_id,
                    )
                    if products:
                        aggregated_products.extend(products)
                    continue
                if intent == IntentType.SHOW_NEARBY_PHARMACIES:
                    _log_platform_call("list_pharmacies", action.parameters)
                    pharmacies = self._platform_client.list_pharmacies(action.parameters, trace_id=trace_id)
                    if pharmacies:
                        platform_data.pharmacies = pharmacies
                    continue
                if intent in {IntentType.FIND_RECOMMENDATION, IntentType.FIND_POPULAR, IntentType.FIND_NEW}:
                    _log_platform_call("get_recommendations", {"user_id": user_id, "context": action.parameters})
                    recommendations = self._platform_client.get_recommendations(
                        user_id=user_id or None,
                        context=action.parameters,
                        trace_id=trace_id,
                    )
                    if recommendations:
                        platform_data.recommendations = recommendations
                    continue
                if intent in DATA_DRIVEN_NAV_INTENTS:
                    executed_data_intents.add(intent)
                if intent == IntentType.ADD_TO_CART:
                    _log_platform_call("add_to_cart", action.parameters)
                    cart_snapshot = await self._platform_client.add_to_cart(action.parameters, request, trace_id=trace_id)
                    if cart_snapshot:
                        platform_data.cart = cart_snapshot
                    # Не вызываем show_cart после add_to_cart — add_to_cart уже возвращает 
                    # актуальную корзину с message об успешном добавлении
                    continue
                if intent in DATA_DRIVEN_NAV_INTENTS:
                    await _fulfill_data_intent(intent, action.parameters)
                    continue
                if intent == IntentType.ADD_TO_FAVORITES:
                    if not user_id:
                        logger.info("ADD_TO_FAVORITES requested without user_id; skipping.")
                        continue
                    product_id = action.parameters.get("product_id")
                    if not product_id:
                        logger.info("ADD_TO_FAVORITES missing product_id.")
                        continue
                    _log_platform_call("add_favorite", {"user_id": user_id, "product_id": product_id})
                    platform_data.favorites = self._platform_client.add_favorite(user_id, product_id, trace_id=trace_id)
                    continue
                if intent == IntentType.REMOVE_FROM_FAVORITES:
                    if not user_id:
                        logger.info("REMOVE_FROM_FAVORITES requested without user_id; skipping.")
                        continue
                    product_id = action.parameters.get("product_id")
                    if not product_id:
                        logger.info("REMOVE_FROM_FAVORITES missing product_id.")
                        continue
                    _log_platform_call("remove_favorite", {"user_id": user_id, "product_id": product_id})
                    platform_data.favorites = self._platform_client.remove_favorite(user_id, product_id, trace_id=trace_id)
                    continue
                if intent == IntentType.SHOW_ACTIVE_ORDERS:
                    if not user_id:
                        logger.info("SHOW_ACTIVE_ORDERS requested without user_id; skipping.")
                        continue
                    _log_platform_call("get_orders", {"user_id": user_id, "status": "active"})
                    orders = self._platform_client.get_orders(user_id, status="active", trace_id=trace_id)
                    _append_orders(orders)
                    continue
                if intent == IntentType.SHOW_COMPLETED_ORDERS:
                    if not user_id:
                        logger.info("SHOW_COMPLETED_ORDERS requested without user_id; skipping.")
                        continue
                    _log_platform_call("get_orders", {"user_id": user_id, "status": "completed"})
                    orders = self._platform_client.get_orders(user_id, status="completed", trace_id=trace_id)
                    _append_orders(orders)
                    continue
                _log_platform_call(
                    "dispatch",
                    {
                        "intent": getattr(intent, "value", intent) if intent else None,
                        "action_type": getattr(action.type, "value", action.type),
                    },
                )
                result = await self._platform_client.dispatch(
                    action,
                    request,
                    user_profile=user_profile,
                    trace_id=trace_id,
                )
                platform_data.merge(result)
            elif action.type == ActionType.NEED_AUTH:
                additional_notes.append("Нужно авторизоваться, чтобы продолжить действие.")
            elif action.type == ActionType.NOOP:
                logger.debug("NOOP action encountered; skipping.")
            elif action.type == ActionType.SHOW_UI_HINT:
                if action.intent and action.intent in DATA_DRIVEN_NAV_INTENTS:
                    pending_data_intents.add(action.intent)
                continue

        if navigation_mode:
            has_hint = any(action.type == ActionType.SHOW_UI_HINT for action in actions)
            if not has_hint:
                hint_parameters: Dict[str, Any] = {}
                if top_intent:
                    hint_parameters["target_intent"] = top_intent.value
                actions.append(
                    AssistantAction(
                        type=ActionType.SHOW_UI_HINT,
                        intent=top_intent,
                        parameters=hint_parameters,
                    )
                )

        if aggregated_products:
            unique_products = _deduplicate_products(aggregated_products)
            platform_data.products = unique_products[:PRODUCT_RESULT_LIMIT]

        for intent in pending_data_intents - executed_data_intents:
            await _fulfill_data_intent(intent, {})

        async def _enrich_profile_snapshot() -> None:
            if not user_id or not platform_data.user_profile:
                return
            if not platform_data.orders:
                _log_platform_call("get_orders", {"user_id": user_id, "status": "active", "stage": "enrich_profile"})
                active_orders = self._platform_client.get_orders(user_id, status="active", trace_id=trace_id)
                _append_orders(active_orders)
            if not platform_data.orders:
                _log_platform_call(
                    "get_orders", {"user_id": user_id, "status": "completed", "stage": "enrich_profile"}
                )
                completed_orders = self._platform_client.get_orders(user_id, status="completed", trace_id=trace_id)
                _append_orders(completed_orders[:2])
            if not platform_data.favorites:
                _log_platform_call("get_favorites", {"user_id": user_id, "stage": "enrich_profile"})
                platform_data.favorites = self._platform_client.get_favorites(user_id, trace_id=trace_id)
            if platform_data.cart is None:
                _log_platform_call("show_cart", {"reason": "enrich_profile"})
                cart_snapshot = await self._platform_client.show_cart({}, request, trace_id=trace_id)
                if cart_snapshot:
                    platform_data.cart = cart_snapshot

        await _enrich_profile_snapshot()

        if not platform_data.recommendations and user_id:
            platform_data.recommendations = self._platform_client.get_recommendations(
                user_id=user_id,
                context={"reason": "recent_purchases_fallback"},
                trace_id=trace_id,
            )

        reply = assistant_response.reply
        if additional_notes:
            text = f"{reply.text}\n\n" + " ".join(additional_notes)
            reply = Reply(text=text, display_hints=reply.display_hints)

        def ensure_quick_replies() -> None:
            if not meta.quick_replies:
                meta.quick_replies = default_quick_replies()

        if skip_reason == "unknown_intent":
            reply_text = reply.text.strip() or UNKNOWN_FALLBACK_REPLY
            if "не понял" not in reply_text.lower():
                reply_text = UNKNOWN_FALLBACK_REPLY + " " + reply_text
            reply = Reply(text=reply_text.strip(), display_hints=reply.display_hints)
            ensure_quick_replies()
        elif skip_reason == "small_talk":
            reply_text = reply.text.strip() or SMALL_TALK_FALLBACK_REPLY
            reply = Reply(text=reply_text, display_hints=reply.display_hints)
            ensure_quick_replies()
        elif skip_reason == "low_confidence":
            reply_text = reply.text.strip() or LOW_CONFIDENCE_FALLBACK
            clarifier = " Можете уточнить, что именно нужно из ассортимента аптеки?"
            if "?" not in reply_text:
                reply_text = reply_text.rstrip(".! ") + "?"
            reply_text = f"{reply_text}{clarifier}"
            reply = Reply(text=reply_text, display_hints=reply.display_hints)
            ensure_quick_replies()

        if needs_symptom_clarification:
            reply = Reply(
                text=_build_symptom_clarification_text(reply.text, symptom_missing_fields),
                display_hints=reply.display_hints,
            )
            symptom_replies = _symptom_clarification_quick_replies(top_intent)
            meta.quick_replies = _merge_quick_replies(meta.quick_replies, symptom_replies)

        if product_query_attempted and not platform_data.products:
            reply = Reply(text=_no_results_reply_text(reply.text), display_hints=reply.display_hints)
            quick_replies = _no_results_quick_replies()
            if meta.quick_replies:
                quick_replies.extend(meta.quick_replies)
            meta.quick_replies = quick_replies

        if platform_data.orders:
            self._append_purchase_history_replies(meta, platform_data.orders)
            if user_id:
                self._maybe_flag_loyal_customer(user_id, platform_data.orders)

        # Используем message из cart если есть (для ADD_TO_CART, SHOW_CART и т.д.)
        cart_message = None
        if platform_data.cart and isinstance(platform_data.cart, dict):
            cart_message = platform_data.cart.get("message")
        
        if cart_message:
            # Сообщение о действии с корзиной имеет приоритет
            reply = Reply(text=cart_message, display_hints=reply.display_hints)
        elif platform_data.message:
            # Для правовых интентов статический ответ имеет приоритет
            reply = Reply(text=platform_data.message, display_hints=reply.display_hints)

        # Сначала beautify_reply для улучшения текста LLM-ом (если включено и LLM доступен)
        llm_available = bool(self._assistant_client._langchain_client) and bool(self._settings.openai_api_key)
        if self._settings.enable_beautify_reply and llm_available:
            try:
                reply = await self._assistant_client.beautify_reply(
                    reply=reply,
                    data=platform_data,
                    constraints={
                        "style": "neutral_helpful",
                        "avoid_medical_advice": True,
                        "max_length": 600,
                    },
                    user_message=request.message,
                    conversation_id=conversation_id,
                    user_id=request.user_id,
                    intent=getattr(top_intent, "value", top_intent) if top_intent else None,
                    router_matched=router_matched,
                    slot_filling_used=slot_filling_used,
                    channel=getattr(channel, "value", channel) if channel else None,
                    trace_metadata={
                        "actions": [
                            getattr(action.intent, "value", action.intent)
                            for action in actions
                            if action.intent
                        ],
                        "llm_backend": llm_backend,
                        "llm_used": llm_used,
                        "trace_id": trace_id,  # Включаем trace_id в metadata
                    },
                    debug_builder=builder,
                )
            except Exception as exc:  # pragma: no cover - safety net
                logger.warning("Beautify reply failed, using original. error=%s", exc)
        # SafetyFilter ПОСЛЕ beautify_reply для гарантии безопасности финального ответа
        reply = Reply(text=SafetyFilter.sanitize_reply(reply.text), display_hints=reply.display_hints)
        meta = SafetyFilter.ensure_disclaimer(meta)
        metrics_snapshot = get_metrics_service().snapshot()
        builder.add_extra("metrics_snapshot", metrics_snapshot.__dict__)
        builder.set_slot_filling_used(slot_filling_used)
        builder.set_router_matched(router_matched)
        if llm_backend:
            builder.add_extra("llm_backend", llm_backend if llm_used else None)

        # Apply price filter to recommendations if provided in actions
        price_limit: float | None = None
        for action in actions:
            params = action.parameters or {}
            price_limit = self._extract_price_from_params(params)
            if price_limit is not None:
                break
        if price_limit is not None and platform_data.recommendations:
            platform_data.recommendations = [
                rec
                for rec in platform_data.recommendations
                if rec.get("price") is None or rec.get("price") <= price_limit
            ]

        # Finalize source if not set explicitly
        debug_snapshot = builder.build()
        if not debug_snapshot.get("source"):
            source = None
            if debug_snapshot.get("router_matched"):
                source = "router+slots" if debug_snapshot.get("slot_filling_used") else "router"
            elif debug_snapshot.get("slot_filling_used"):
                source = "slots"
            elif debug_snapshot.get("llm_used"):
                source = "llm+platform" if platform_data.has_content() else "llm"
            if source:
                builder.set_source(source)
                debug_snapshot["source"] = source

        meta.debug = builder.build()

        self._remember_dialog_state(
            conversation_id=conversation_id,
            skip_reason=skip_reason,
            intent=top_intent,
            actions=actions,
            platform_data=platform_data,
            reply=reply,
        )

        return ChatResponse(
            conversation_id=conversation_id,
            reply=reply,
            actions=actions,
            meta=meta,
            data=platform_data,
            ui_state=request.ui_state.model_dump() if request.ui_state else None,
        )

    @staticmethod
    def _infer_llm_context(assistant_response: AssistantResponse) -> tuple[bool, str | None]:
        meta = assistant_response.meta
        debug = (meta.debug or {}) if meta and meta.debug else {}
        llm_used = debug.get("llm_used")
        if llm_used is None:
            llm_used = True
        backend = debug.get("llm_backend")
        if llm_used and backend is None:
            backend = "langchain"
        if not llm_used:
            backend = None
        return bool(llm_used), backend

    def _append_purchase_history_replies(self, meta: AssistantMeta, orders: List[Dict[str, Any]]) -> None:
        suggestions = build_purchase_history_quick_replies(orders)
        if not suggestions:
            return
        if meta.quick_replies is None:
            meta.quick_replies = []
        meta.quick_replies.extend(suggestions)

    def _maybe_flag_loyal_customer(self, user_id: str, orders: List[Dict[str, Any]]) -> None:
        if not user_id:
            return
        unique_orders = {order.get("order_id") for order in orders if order.get("order_id")}
        if len(unique_orders) >= 3:
            self._user_profile_store.add_tag(user_id, "loyal_customer")

    def _remember_dialog_state(
        self,
        *,
        conversation_id: str,
        skip_reason: str | None,
        intent: IntentType | None,
        actions: List[AssistantAction],
        platform_data: DataPayload,
        reply: Reply,
    ) -> None:
        if not conversation_id:
            return
        if skip_reason in {"unknown_intent", "small_talk"}:
            self._dialog_state_store.clear_state(conversation_id)
            return
        combined_slots: Dict[str, Any] = {}
        for action in actions:
            combined_slots.update(action.parameters)
        channel = None
        if actions:
            channel = self._resolve_channel(actions[0])
        self._dialog_state_store.upsert_state(
            conversation_id,
            current_intent=intent,
            channel=channel,
            slots=combined_slots,
            context_products=platform_data.products,
            last_reply=reply.text,
        )

    def _capture_preferences_from_actions(self, user_id: str, actions: List[AssistantAction]) -> None:
        """Persist explicit numeric preferences extracted by LLM actions."""

        updates: Dict[str, Any] = {}
        for action in actions:
            params = action.parameters or {}
            age = self._extract_age_from_params(params)
            if age is not None:
                updates["age"] = age
            price = self._extract_price_from_params(params)
            if price is not None:
                updates["default_max_price"] = price
        if updates:
            self._user_profile_store.update_preferences(user_id, **updates)

    def _extract_age_from_params(self, params: Dict[str, Any]) -> int | None:
        if "age" not in params:
            return None
        raw = params.get("age")
        try:
            age = int(raw)
        except (TypeError, ValueError):
            return None
        if 0 < age <= 110:
            return age
        return None

    def _extract_price_from_params(self, params: Dict[str, Any]) -> float | None:
        price_keys = ("price_max", "max_price", "budget", "default_max_price")
        upper_bound = getattr(self._settings, "price_max_upper_bound", None)
        for key in price_keys:
            if key not in params:
                continue
            raw = params.get(key)
            try:
                price = float(raw)
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue
            if upper_bound is not None and price > upper_bound:
                price = upper_bound
            return price
        return None

    def _resolve_channel(self, action: AssistantAction) -> ActionChannel:
        if action.channel:
            try:
                return ActionChannel(action.channel)
            except ValueError:
                logger.debug("Unknown action channel %s; defaulting to OTHER", action.channel)
        intent = action.intent
        if not intent:
            return ActionChannel.OTHER
        category = get_intent_category(intent)
        if category == "navigation":
            return ActionChannel.NAVIGATION
        if category == "order":
            return ActionChannel.ORDER
        if category == "symptom":
            return ActionChannel.DATA
        return ActionChannel.OTHER

    def _handle_support_action(
        self,
        *,
        action: AssistantAction,
        request: ChatRequest,
        user_profile: UserProfile | None,
        platform_data: DataPayload,
    ) -> Dict[str, Any] | None:
        issue = action.parameters.get("issue") or request.message
        try:
            return self._platform_client.create_support_ticket(
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                issue=issue,
                payload={
                    "intent": getattr(action.intent, "value", action.intent),
                    "parameters": action.parameters,
                    "orders": platform_data.orders,
                    "cart": platform_data.cart,
                    "user_profile": user_profile.model_dump() if user_profile else None,
                },
            )
        except Exception as exc:  # pragma: no cover - fail safe
            logger.exception("Failed to escalate to support: %s", exc)
            return None

