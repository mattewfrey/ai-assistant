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
from .safety_filter import SafetyFilter
from .slot_manager import SlotManager
from .router import RouterResult
from .static_responses import get_legal_response, is_legal_intent
from .user_profile_store import UserProfileStore, get_user_profile_store

logger = logging.getLogger(__name__)

UNKNOWN_FALLBACK_REPLY = "Я пока не понял запрос. Могу помочь с подбором препаратов, заказами или корзиной."
LOW_CONFIDENCE_FALLBACK = "Кажется, я не до конца понял запрос. Можете уточнить, что именно нужно?"
SMALL_TALK_FALLBACK_REPLY = "Всегда рад помочь! Если потребуется что-то из аптеки, просто скажите."

DEFAULT_QUICK_REPLY_PRESETS: List[Dict[str, Any]] = [
    {
        "label": "Подбор по симптомам",
        "query": "Подбери препарат по моим симптомам",
        "intent": IntentType.FIND_BY_SYMPTOM.value,
        "parameters": {},
    },
    {
        "label": "Помощь по заказу",
        "query": "Покажи статус моего заказа",
        "intent": IntentType.SHOW_ORDER_STATUS.value,
        "parameters": {},
    },
    {
        "label": "Показать корзину",
        "query": "Покажи мою корзину",
        "intent": IntentType.SHOW_CART.value,
        "parameters": {},
    },
    {
        "label": "Выбрать аптеку",
        "query": "Покажи ближайшие аптеки",
        "intent": IntentType.SHOW_NEARBY_PHARMACIES.value,
        "parameters": {},
    },
]


def _default_quick_replies() -> List[Dict[str, Any]]:
    return [preset.copy() for preset in DEFAULT_QUICK_REPLY_PRESETS]


PRODUCT_RESULT_LIMIT = 8

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


_SYMPTOM_PARAM_ALIASES: Dict[str, Sequence[str]] = {
    "age": ("age", "age_years", "age_group"),
}

_SYMPTOM_FIELD_QUESTIONS: Dict[str, str] = {
    "age": "Сколько лет человеку, для которого подбираем лечение?",
}

_SYMPTOM_EXTRA_HINT = (
    "Также уточните температуру, длительность симптомов и хронические заболевания, чтобы подобрать безопасные товары."
)


def _has_non_empty_value(parameters: Dict[str, Any], keys: Sequence[str]) -> bool:
    for key in keys:
        if key not in parameters:
            continue
        value = parameters[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return True
    return False


def _missing_symptom_parameters(parameters: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if not _has_non_empty_value(parameters, _SYMPTOM_PARAM_ALIASES["age"]):
        missing.append("age")
    return missing


def _build_symptom_clarification_text(previous_text: str | None, missing_fields: Sequence[str]) -> str:
    parts: List[str] = []
    if previous_text:
        stripped = previous_text.strip()
        if stripped:
            parts.append(stripped)
    for field in missing_fields:
        question = _SYMPTOM_FIELD_QUESTIONS.get(field)
        if question:
            parts.append(question)
    parts.append(_SYMPTOM_EXTRA_HINT)
    return " ".join(part.strip() for part in parts if part).strip()


def _symptom_clarification_quick_replies(intent: IntentType | None) -> List[Dict[str, Any]]:
    if not intent:
        return []
    intent_value = intent.value
    return [
        {
            "label": "Ребёнок до 6 лет",
            "query": "Нам 5 лет",
            "intent": intent_value,
            "parameters": {"age": 5},
        },
        {
            "label": "Подросток 12-17",
            "query": "Возраст 14 лет",
            "intent": intent_value,
            "parameters": {"age": 14},
        },
        {
            "label": "Взрослый 18+",
            "query": "Мне 30 лет",
            "intent": intent_value,
            "parameters": {"age": 30},
        },
    ]


def _merge_quick_replies(
    existing: List[Dict[str, Any]] | None, additions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not additions:
        return existing or []
    merged = [reply.copy() for reply in additions]
    if existing:
        merged.extend(existing)
    return merged


def _no_results_quick_replies() -> List[Dict[str, Any]]:
    return [
        {
            "label": "Показать популярные категории",
            "query": "Покажи популярные категории",
            "intent": IntentType.FIND_BY_CATEGORY.value,
            "parameters": {"category": "popular"},
        },
        {
            "label": "Подобрать без фильтров",
            "query": "Подбери препараты заново",
            "intent": IntentType.FIND_BY_SYMPTOM.value,
            "parameters": {},
        },
        {
            "label": "Сменить аптеку",
            "query": "Покажи другие аптеки",
            "intent": IntentType.SHOW_NEARBY_PHARMACIES.value,
            "parameters": {},
        },
    ]


def _deduplicate_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_ids: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for product in products:
        product_id = product.get("id")
        if product_id and product_id in seen_ids:
            continue
        if product_id:
            seen_ids.add(product_id)
        unique.append(product)
    return unique


def _no_results_reply_text(previous_text: str | None) -> str:
    base = "Пока не нашёл подходящих товаров с учётом выбранной аптеки и фильтров."
    suffix = " Могу показать популярные категории или подобрать альтернативы."
    if previous_text:
        return f"{base} {previous_text.strip()}"
    return f"{base}{suffix}"


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
        user_profile: UserProfile | None = None
        user_id = (request.user_id or "").strip()
        if user_id:
            user_profile = self._user_profile_store.get_or_create(user_id)

        actions = list(assistant_response.actions)
        top_intent = actions[0].intent if actions and actions[0].intent else None
        channel = self._resolve_channel(actions[0]) if actions else None
        if meta and meta.debug:
            slot_filling_used = bool(meta.debug.get("slot_filling_used", False))
            builder.set_slot_filling_used(slot_filling_used)
            if "slot_prompt_pending" in meta.debug:
                builder.set_pending_slots(bool(meta.debug.get("slot_prompt_pending")))
        builder.set_trace_id(trace_id)
        builder.add_intent(getattr(top_intent, "value", None))
        log_info(
            logger,
            "Building chat response",
            trace_id=trace_id,
            user_id=request.user_id,
            conversation_id=conversation_id,
            intent=top_intent,
        )
        top_category = get_intent_category(top_intent)
        symptom_missing_fields: List[str] = []
        if top_category == "symptom" and actions:
            symptom_missing_fields = _missing_symptom_parameters(actions[0].parameters)
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

        async def _fulfill_data_intent(intent: IntentType, parameters: Dict[str, Any]) -> None:
            if intent == IntentType.SHOW_CART:
                cart_snapshot = await self._platform_client.show_cart(parameters, request)
                if cart_snapshot:
                    platform_data.cart = cart_snapshot
                return
            if intent == IntentType.SHOW_PROFILE:
                if not user_id:
                    logger.info("SHOW_PROFILE requested without user_id; skipping.")
                    return
                profile = self._platform_client.get_user_profile(user_id)
                addresses = self._platform_client.get_user_addresses(user_id)
                if profile:
                    platform_data.user_profile = profile
                if addresses:
                    platform_data.user_addresses = addresses
                return
            if intent == IntentType.SHOW_FAVORITES:
                if not user_id:
                    logger.info("SHOW_FAVORITES requested without user_id; skipping.")
                    return
                platform_data.favorites = self._platform_client.get_favorites(user_id)

        for action in actions:
            logger.info("Processing action type=%s intent=%s", action.type, action.intent)
            action_channel = self._resolve_channel(action)
            builder.add_intent(getattr(action.intent, "value", action.intent) if action.intent else None)
            log_info(
                logger,
                "Processing action",
                trace_id=trace_id,
                user_id=request.user_id,
                conversation_id=conversation_id,
                intent=action.intent,
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
                    products = await self._platform_client.fetch_products(
                        intent,
                        action.parameters,
                        request,
                        user_profile=user_profile,
                    )
                    if products:
                        aggregated_products.extend(products)
                    continue
                if intent == IntentType.SHOW_NEARBY_PHARMACIES:
                    pharmacies = self._platform_client.list_pharmacies(action.parameters)
                    if pharmacies:
                        platform_data.pharmacies = pharmacies
                    continue
                if intent in {IntentType.FIND_RECOMMENDATION, IntentType.FIND_POPULAR, IntentType.FIND_NEW}:
                    recommendations = self._platform_client.get_recommendations(
                        user_id=user_id or None,
                        context=action.parameters,
                    )
                    if recommendations:
                        platform_data.recommendations = recommendations
                    continue
                if intent in DATA_DRIVEN_NAV_INTENTS:
                    executed_data_intents.add(intent)
                if intent == IntentType.ADD_TO_CART:
                    cart_snapshot = await self._platform_client.add_to_cart(action.parameters, request)
                    if cart_snapshot:
                        platform_data.cart = cart_snapshot
                    if action.parameters.get("refresh_cart", True):
                        refreshed_cart = await self._platform_client.show_cart(action.parameters, request)
                        if refreshed_cart:
                            platform_data.cart = refreshed_cart
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
                    platform_data.favorites = self._platform_client.add_favorite(user_id, product_id)
                    continue
                if intent == IntentType.REMOVE_FROM_FAVORITES:
                    if not user_id:
                        logger.info("REMOVE_FROM_FAVORITES requested without user_id; skipping.")
                        continue
                    product_id = action.parameters.get("product_id")
                    if not product_id:
                        logger.info("REMOVE_FROM_FAVORITES missing product_id.")
                        continue
                    platform_data.favorites = self._platform_client.remove_favorite(user_id, product_id)
                    continue
                if intent == IntentType.SHOW_ACTIVE_ORDERS:
                    if not user_id:
                        logger.info("SHOW_ACTIVE_ORDERS requested without user_id; skipping.")
                        continue
                    orders = self._platform_client.get_orders(user_id, status="active")
                    _append_orders(orders)
                    continue
                if intent == IntentType.SHOW_COMPLETED_ORDERS:
                    if not user_id:
                        logger.info("SHOW_COMPLETED_ORDERS requested without user_id; skipping.")
                        continue
                    orders = self._platform_client.get_orders(user_id, status="completed")
                    _append_orders(orders)
                    continue
                result = await self._platform_client.dispatch(action, request, user_profile=user_profile)
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
                active_orders = self._platform_client.get_orders(user_id, status="active")
                _append_orders(active_orders)
            if not platform_data.orders:
                completed_orders = self._platform_client.get_orders(user_id, status="completed")
                _append_orders(completed_orders[:2])
            if not platform_data.favorites:
                platform_data.favorites = self._platform_client.get_favorites(user_id)
            if platform_data.cart is None:
                cart_snapshot = await self._platform_client.show_cart({}, request)
                if cart_snapshot:
                    platform_data.cart = cart_snapshot

        await _enrich_profile_snapshot()

        if not platform_data.recommendations and user_id:
            platform_data.recommendations = self._platform_client.get_recommendations(
                user_id=user_id,
                context={"reason": "recent_purchases_fallback"},
            )

        reply = assistant_response.reply
        if additional_notes:
            text = f"{reply.text}\n\n" + " ".join(additional_notes)
            reply = Reply(text=text, display_hints=reply.display_hints)

        def ensure_quick_replies() -> None:
            if not meta.quick_replies:
                meta.quick_replies = _default_quick_replies()

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

        # Если есть статический ответ (legal/policy), используем его
        if platform_data.message:
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
                    },
                    debug_builder=builder,
                    trace_id=trace_id,
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
        suggestions = self._build_purchase_history_quick_replies(orders)
        if not suggestions:
            return
        if meta.quick_replies is None:
            meta.quick_replies = []
        meta.quick_replies.extend(suggestions)

    def _build_purchase_history_quick_replies(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not orders:
            return []
        quick_replies: List[Dict[str, Any]] = []
        last_order = orders[0]
        last_items = last_order.get("items") or []
        if last_items:
            quick_replies.append(
                {
                    "label": "Повторить последний заказ",
                    "query": "Повтори мой последний заказ",
                    "intent": None,
                    "parameters": {"order_id": last_order.get("order_id")},
                }
            )
        frequent_item = self._most_frequent_item(orders)
        if frequent_item:
            item_name = frequent_item.get("title") or "ваши частые покупки"
            quick_replies.append(
                {
                    "label": "Показать частые покупки",
                    "query": "Показать товары, которые я часто покупаю",
                    "intent": IntentType.FIND_PRODUCT_BY_NAME.value,
                    "parameters": {
                        "product_id": frequent_item.get("product_id"),
                        "name": item_name,
                    },
                }
            )
        return quick_replies

    def _most_frequent_item(self, orders: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        stats: Dict[str, Dict[str, Any]] = {}
        for order in orders:
            for item in order.get("items") or []:
                product_id = item.get("product_id")
                if not product_id:
                    continue
                bucket = stats.setdefault(product_id, {"count": 0, "title": item.get("title")})
                bucket["count"] += 1
                if not bucket.get("title") and item.get("title"):
                    bucket["title"] = item["title"]
        if not stats:
            return None
        product_id, aggregates = max(stats.items(), key=lambda entry: entry[1]["count"])
        if aggregates["count"] < 2:
            return None
        return {"product_id": product_id, "title": aggregates.get("title")}

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

