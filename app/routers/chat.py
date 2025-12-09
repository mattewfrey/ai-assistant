from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, status

from ..config import Settings, get_settings
from ..intents import ActionChannel, ActionType, IntentType, get_intent_category
from ..models import AssistantAction, AssistantMeta, AssistantResponse, ChatRequest, ChatResponse, Reply
from ..services.assistant_client import AssistantClient
from ..services.conversation_store import get_conversation_store
from ..services.dialog_state_store import get_dialog_state_store
from ..services.debug_meta import DebugMetaBuilder
from ..services.errors import BadRequestError
from ..services.metrics import get_metrics_service
from ..services.local_router import LocalRouterResult, route as local_route
from ..services.orchestrator import Orchestrator
from ..services.platform_client import PlatformApiClient
from ..services.router import RouterService, get_router_service
from ..services.slot_manager import SlotManager, get_slot_manager
from ..services.user_profile_store import UserProfileStore, get_user_profile_store
from ..utils.logging import get_request_logger

router = APIRouter(prefix="/api/ai/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def get_assistant_client(settings: Settings = Depends(get_settings)) -> AssistantClient:
    return AssistantClient(
        settings=settings,
        user_profile_store=get_user_profile_store(),
        dialog_state_store=get_dialog_state_store(),
    )


def get_platform_client(settings: Settings = Depends(get_settings)) -> PlatformApiClient:
    return PlatformApiClient(settings=settings)


def get_orchestrator(
    platform_client: PlatformApiClient = Depends(get_platform_client),
    assistant_client: AssistantClient = Depends(get_assistant_client),
    settings: Settings = Depends(get_settings),
) -> Orchestrator:
    return Orchestrator(
        platform_client=platform_client,
        assistant_client=assistant_client,
        settings=settings,
        user_profile_store=get_user_profile_store(),
        dialog_state_store=get_dialog_state_store(),
    )


def get_user_profile_store_dependency() -> UserProfileStore:
    return get_user_profile_store()


@router.post("/message", response_model=ChatResponse)
async def post_message(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    assistant_client: AssistantClient = Depends(get_assistant_client),
    orchestrator: Orchestrator = Depends(get_orchestrator),
    router_service: RouterService = Depends(get_router_service),
    slot_manager: SlotManager = Depends(get_slot_manager),
    user_profile_store: UserProfileStore = Depends(get_user_profile_store_dependency),
) -> ChatResponse:
    start_time = time.perf_counter()
    metrics = get_metrics_service()
    
    if not request.message.strip():
        raise BadRequestError(
            "message must not be empty",
            reason="empty_message",
            http_status=status.HTTP_400_BAD_REQUEST,
        )

    conversation_id = request.conversation_id or str(uuid4())
    incoming_trace_id = getattr(request, "trace_id", None)
    trace_id = incoming_trace_id or (uuid4().hex if settings.enable_request_tracing else None)
    debug_builder = DebugMetaBuilder(request_id=conversation_id)
    debug_builder.set_trace_id(trace_id)
    normalized_request = request.model_copy(update={"conversation_id": conversation_id, "trace_id": trace_id})
    request_logger = get_request_logger(
        logger,
        trace_id=trace_id,
        user_id=normalized_request.user_id,
        conversation_id=conversation_id,
    )

    request_logger.info("Incoming chat message text=%s", normalized_request.message)

    user_profile = user_profile_store.get_or_create(normalized_request.user_id) if normalized_request.user_id else None
    dialog_state_store = get_dialog_state_store()
    dialog_state = dialog_state_store.get_state(conversation_id)

    # Slot follow-up handling comes first
    followup_decision = slot_manager.try_handle_followup(
        request_message=normalized_request.message,
        conversation_id=conversation_id,
        user_profile=user_profile,
        debug_builder=debug_builder,
        trace_id=trace_id,
    )
    if followup_decision.handled and followup_decision.assistant_response:
        if followup_decision.assistant_response.actions:
            metrics.record_slot_success()
        else:
            metrics.record_slot_prompt()
        chat_response = await orchestrator.build_response(
            request=normalized_request,
            assistant_response=followup_decision.assistant_response,
            router_matched=True,
            debug_builder=debug_builder,
            trace_id=trace_id,
        )
        _record_history(conversation_id, normalized_request.message, chat_response.reply.text)
        _record_latency(start_time)
        return chat_response

    if settings.enable_local_router:
        local_result = local_route(normalized_request)
        if local_result.matched:
            metrics.record_router_match(True)
            assistant_response = _build_local_assistant_response(local_result)
            debug_builder.set_router_matched(True).set_llm_used(False).set_source("local_router").add_intent(
                getattr(local_result.intent, "value", None)
            )
            chat_response = await orchestrator.build_response(
                request=normalized_request,
                assistant_response=assistant_response,
                router_matched=True,
                debug_builder=debug_builder,
                trace_id=trace_id,
            )
            _record_history(conversation_id, normalized_request.message, chat_response.reply.text)
            _record_latency(start_time)
            return chat_response

    router_result = router_service.match(
        request=normalized_request,
        user_profile=user_profile,
        dialog_state=dialog_state,
        debug_builder=debug_builder,
        trace_id=trace_id,
    )
    if router_result.matched:
        metrics.record_router_match(True)
        chat_response = await orchestrator.build_response_from_router(
            request=normalized_request,
            router_result=router_result,
            slot_manager=slot_manager,
            conversation_id=conversation_id,
            user_profile=user_profile,
            debug_builder=debug_builder,
            trace_id=trace_id,
        )
        _record_history(conversation_id, normalized_request.message, chat_response.reply.text)
        _record_latency(start_time)
        return chat_response

    metrics.record_router_match(False)

    # Rate limiting check before LLM call
    if not metrics.check_rate_limit(
        user_id=normalized_request.user_id,
        window_seconds=settings.llm_rate_limit_window_seconds,
        max_calls=settings.llm_rate_limit_max_calls,
    ):
        request_logger.warning("Rate limit exceeded")
        raise BadRequestError(
            "Превышен лимит запросов. Пожалуйста, подождите немного.",
            reason="rate_limit_exceeded",
            http_status=status.HTTP_429_TOO_MANY_REQUESTS,
        )

    intents = [intent.value for intent in IntentType]
    assistant_response = await assistant_client.analyze_message(
        normalized_request, intents, debug_builder=debug_builder, trace_id=trace_id
    )

    action_log_payload = [
        {
            "type": getattr(action.type, "value", action.type),
            "intent": getattr(action.intent, "value", action.intent),
            "parameters": action.parameters,
        }
        for action in assistant_response.actions
    ]
    request_logger.info("Assistant actions selected=%s", action_log_payload)

    chat_response = await orchestrator.build_response(
        request=normalized_request,
        assistant_response=assistant_response,
        debug_builder=debug_builder,
        trace_id=trace_id,
    )
    _record_latency(start_time)
    return chat_response


def _build_local_assistant_response(result: LocalRouterResult) -> AssistantResponse:
    intent = result.intent
    parameters = result.parameters or {}
    channel = _resolve_channel_from_intent(intent)
    reply_text = _reply_text_for_intent(intent, parameters)
    meta = AssistantMeta(
        top_intent=getattr(intent, "value", None),
        confidence=0.99,
        debug={"source": "local_router", "llm_used": False, "router_matched": True},
    )
    action = AssistantAction(
        type=ActionType.CALL_PLATFORM_API,
        intent=intent,
        channel=channel,
        parameters=parameters,
    )
    return AssistantResponse(
        reply=Reply(text=reply_text),
        actions=[action],
        meta=meta,
    )


def _resolve_channel_from_intent(intent: IntentType | None) -> ActionChannel | None:
    category = get_intent_category(intent)
    if category == "navigation":
        return ActionChannel.NAVIGATION
    if category == "order":
        return ActionChannel.ORDER
    if category == "symptom":
        return ActionChannel.DATA
    if intent == IntentType.FIND_PRODUCT_BY_NAME:
        return ActionChannel.DATA
    return ActionChannel.OTHER


def _reply_text_for_intent(intent: IntentType | None, parameters: dict[str, Any]) -> str:
    if intent == IntentType.SHOW_CART:
        return "Показываю вашу корзину."
    if intent in (IntentType.SHOW_ORDER_HISTORY, IntentType.SHOW_ACTIVE_ORDERS):
        return "Открываю ваши заказы."
    if intent == IntentType.SHOW_FAVORITES:
        return "Показываю избранные товары."
    if intent == IntentType.SHOW_PROFILE:
        return "Открываю профиль."
    if intent == IntentType.FIND_PRODUCT_BY_NAME:
        name = parameters.get("product_name") or parameters.get("name") or "товар"
        return f'Ищу "{name}".'
    return "Понял, выполняю."


def _record_history(conversation_id: str, user_message: str, assistant_text: str | None) -> None:
    store = get_conversation_store()
    store.append_message(conversation_id, "user", user_message)
    if assistant_text:
        store.append_message(conversation_id, "assistant", assistant_text)


def _record_latency(start_time: float) -> None:
    """Record response latency in milliseconds."""
    latency_ms = (time.perf_counter() - start_time) * 1000
    get_metrics_service().record_response_latency(latency_ms)
