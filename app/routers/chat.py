from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from ..config import Settings, get_settings
from ..intents import IntentType
from ..models import ChatRequest, ChatResponse
from ..models.assistant import AssistantMeta
from ..services.assistant_client import AssistantClient, AssistantClientError
from ..services.conversation_store import get_conversation_store
from ..services.dialog_state_store import get_dialog_state_store
from ..services.metrics import get_metrics_service
from ..services.orchestrator import Orchestrator
from ..services.platform_client import PlatformApiClient
from ..services.router import RouterService, get_router_service
from ..services.slot_manager import SlotManager, get_slot_manager
from ..services.user_profile_store import UserProfileStore, get_user_profile_store

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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="message must not be empty")

    conversation_id = request.conversation_id or str(uuid4())
    normalized_request = request.model_copy(update={"conversation_id": conversation_id})

    user_profile = user_profile_store.get_or_create(normalized_request.user_id) if normalized_request.user_id else None
    dialog_state_store = get_dialog_state_store()
    dialog_state = dialog_state_store.get_state(conversation_id)

    # Slot follow-up handling comes first
    followup_decision = slot_manager.try_handle_followup(
        request_message=normalized_request.message,
        conversation_id=conversation_id,
        user_profile=user_profile,
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
        )
        _attach_debug_metadata(
            chat_response,
            router_matched=True,
            slot_filling_used=followup_decision.slot_filling_used,
        )
        _record_history(conversation_id, normalized_request.message, chat_response.reply.text)
        _record_latency(start_time)
        return chat_response

    router_result = router_service.match(
        request=normalized_request,
        user_profile=user_profile,
        dialog_state=dialog_state,
    )
    if router_result.matched:
        metrics.record_router_match(True)
        chat_response = await orchestrator.build_response_from_router(
            request=normalized_request,
            router_result=router_result,
            slot_manager=slot_manager,
            conversation_id=conversation_id,
            user_profile=user_profile,
        )
        _attach_debug_metadata(
            chat_response,
            router_matched=True,
            slot_filling_used=bool(router_result.missing_slots),
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
        logger.warning(
            "Rate limit exceeded for user_id=%s conversation_id=%s",
            normalized_request.user_id,
            conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Превышен лимит запросов. Пожалуйста, подождите немного.",
        )

    intents = [intent.value for intent in IntentType]
    try:
        assistant_response = await assistant_client.analyze_message(normalized_request, intents)
    except AssistantClientError as exc:
        logger.exception("Assistant error")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    action_log_payload = [
        {
            "type": getattr(action.type, "value", action.type),
            "intent": getattr(action.intent, "value", action.intent),
            "parameters": action.parameters,
        }
        for action in assistant_response.actions
    ]
    logger.info("Assistant actions for %s: %s", conversation_id, action_log_payload)

    chat_response = await orchestrator.build_response(
        request=normalized_request,
        assistant_response=assistant_response,
    )
    _attach_debug_metadata(
        chat_response,
        router_matched=False,
        slot_filling_used=False,
    )
    _record_latency(start_time)
    return chat_response


def _record_history(conversation_id: str, user_message: str, assistant_text: str | None) -> None:
    store = get_conversation_store()
    store.append_message(conversation_id, "user", user_message)
    if assistant_text:
        store.append_message(conversation_id, "assistant", assistant_text)


def _record_latency(start_time: float) -> None:
    """Record response latency in milliseconds."""
    latency_ms = (time.perf_counter() - start_time) * 1000
    get_metrics_service().record_response_latency(latency_ms)


def _attach_debug_metadata(
    chat_response: ChatResponse,
    *,
    router_matched: bool,
    slot_filling_used: bool,
) -> None:
    metrics_snapshot = get_metrics_service().snapshot().__dict__
    meta = chat_response.meta or AssistantMeta()
    debug_payload = dict(meta.debug or {})
    debug_payload["router_matched"] = router_matched
    existing_slot_flag = bool(debug_payload.get("slot_filling_used"))
    debug_payload["slot_filling_used"] = existing_slot_flag or slot_filling_used
    debug_payload.setdefault("llm_used", False)
    debug_payload.setdefault("llm_cached", False)
    if "llm_backend" not in debug_payload:
        debug_payload["llm_backend"] = "langchain" if debug_payload["llm_used"] else None
    debug_payload["metrics_snapshot"] = metrics_snapshot
    meta.debug = debug_payload
    chat_response.meta = meta
