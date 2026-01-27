from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request

from ..config import Settings, get_settings
from ..models.product_chat import ProductChatRequest, ProductChatResponse
from ..models.product_faq import ProductFAQResponse
from ..models.proactive_hints import ProactiveHintsRequest, ProactiveHintsResponse, ProactiveTriggerType
from ..models.drug_interaction import DrugInteractionCheckRequest, DrugInteractionCheckResponse
from ..models.smart_analogs import SmartAnalogsRequest, SmartAnalogsResponse
from ..models.course_calculator import CourseCalculatorRequest, CourseCalculatorResult
from ..models.purchase_history import PurchaseHistoryRequest, PurchaseHistoryResponse
from ..services.product_chat_service import ProductChatService
from ..services.drug_interaction_service import DrugInteractionService, get_drug_interaction_service
from ..services.smart_analogs_service import SmartAnalogsService, get_smart_analogs_service
from ..services.course_calculator_service import CourseCalculatorService, get_course_calculator_service
from ..services.purchase_history_service import PurchaseHistoryService, get_purchase_history_service
from ..services.product_context_builder import ProductContextBuilder
from ..services.product_faq_service import ProductFAQService
from ..services.product_gateway_client import ProductGatewayClient
from ..services.proactive_hints_service import ProactiveHintsService
from ..services.product_policy_guard import ProductPolicyGuard
from ..services.product_chat_session_store import ProductChatSessionStore, get_product_chat_session_store
from ..services.langchain_llm import LangchainLLMClient
from ..services.llm_debug_store import get_llm_debug_store

router = APIRouter(prefix="/api/product-ai", tags=["product-ai-chat"])


def get_product_gateway_client(settings: Settings = Depends(get_settings)) -> ProductGatewayClient:
    return ProductGatewayClient(settings=settings)


def get_product_context_builder(
    settings: Settings = Depends(get_settings),
    gateway_client: ProductGatewayClient = Depends(get_product_gateway_client),
) -> ProductContextBuilder:
    return ProductContextBuilder(settings=settings, gateway_client=gateway_client)


def get_product_policy_guard() -> ProductPolicyGuard:
    return ProductPolicyGuard()


def get_product_chat_llm_client(settings: Settings = Depends(get_settings)) -> Optional[LangchainLLMClient]:
    if not settings.openai_api_key:
        return None
    return LangchainLLMClient(settings)


def get_product_chat_service(
    settings: Settings = Depends(get_settings),
    context_builder: ProductContextBuilder = Depends(get_product_context_builder),
    policy_guard: ProductPolicyGuard = Depends(get_product_policy_guard),
    llm_client: Optional[LangchainLLMClient] = Depends(get_product_chat_llm_client),
    session_store: ProductChatSessionStore = Depends(get_product_chat_session_store),
) -> ProductChatService:
    return ProductChatService(
        settings=settings,
        context_builder=context_builder,
        policy_guard=policy_guard,
        llm_client=llm_client,
        session_store=session_store,
    )


def get_product_faq_service(
    settings: Settings = Depends(get_settings),
    context_builder: ProductContextBuilder = Depends(get_product_context_builder),
    llm_client: Optional[LangchainLLMClient] = Depends(get_product_chat_llm_client),
) -> ProductFAQService:
    return ProductFAQService(
        settings=settings,
        context_builder=context_builder,
        llm_client=llm_client,
    )


def get_proactive_hints_service(
    settings: Settings = Depends(get_settings),
    context_builder: ProductContextBuilder = Depends(get_product_context_builder),
) -> ProactiveHintsService:
    return ProactiveHintsService(
        settings=settings,
        context_builder=context_builder,
    )


@router.post(
    "/proactive/hints",
    response_model=ProactiveHintsResponse,
    summary="Get proactive hints for a product based on user behavior",
    description="Returns contextual hints to show to users based on trigger type (exit intent, scroll depth, etc.)",
)
async def get_proactive_hints(
    request: ProactiveHintsRequest,
    http_request: Request,
    settings: Settings = Depends(get_settings),
    service: ProactiveHintsService = Depends(get_proactive_hints_service),
) -> ProactiveHintsResponse:
    """Get proactive hints for a product."""
    trace_id = uuid.uuid4().hex if settings.enable_request_tracing else None
    authorization = http_request.headers.get("Authorization")
    return await service.get_hints(
        request=request,
        authorization=authorization,
        trace_id=trace_id,
    )


@router.get(
    "/faq/{product_id}",
    response_model=ProductFAQResponse,
    summary="Get auto-generated FAQs for a product",
    description="Returns a list of frequently asked questions and answers based on product data.",
)
async def get_product_faqs(
    product_id: str,
    http_request: Request,
    store_id: Optional[str] = Query(None, description="Store ID for availability context"),
    shipping_method: Optional[str] = Query(None, description="Shipping method for delivery context"),
    force_refresh: bool = Query(False, description="Force regeneration of FAQs"),
    settings: Settings = Depends(get_settings),
    service: ProductFAQService = Depends(get_product_faq_service),
) -> ProductFAQResponse:
    """Get auto-generated FAQs for a product."""
    trace_id = uuid.uuid4().hex if settings.enable_request_tracing else None
    authorization = http_request.headers.get("Authorization")
    return await service.get_faqs(
        product_id=product_id,
        store_id=store_id,
        shipping_method=shipping_method,
        authorization=authorization,
        trace_id=trace_id,
        force_refresh=force_refresh,
    )


@router.post("/chat/message", response_model=ProductChatResponse)
async def post_product_message(
    request: ProductChatRequest,
    http_request: Request,
    settings: Settings = Depends(get_settings),
    service: ProductChatService = Depends(get_product_chat_service),
) -> ProductChatResponse:
    trace_id = uuid.uuid4().hex if settings.enable_request_tracing else None
    authorization = http_request.headers.get("Authorization")
    return await service.handle(
        request=request,
        authorization=authorization,
        trace_id=trace_id,
    )


@router.post(
    "/drug-interactions/check",
    response_model=DrugInteractionCheckResponse,
    summary="Check drug interactions",
    description="Check for potential drug interactions between a product and other medications.",
)
async def check_drug_interactions(
    request: DrugInteractionCheckRequest,
    settings: Settings = Depends(get_settings),
) -> DrugInteractionCheckResponse:
    """Check for drug interactions."""
    trace_id = uuid.uuid4().hex if settings.enable_request_tracing else None
    service = get_drug_interaction_service(settings)
    return await service.check_interactions(
        request=request,
        trace_id=trace_id,
    )


@router.post(
    "/analogs/find",
    response_model=SmartAnalogsResponse,
    summary="Find cheaper analogs by INN",
    description="Find cheaper alternatives to a medication based on the same active ingredient (INN/МНН).",
)
async def find_smart_analogs(
    request: SmartAnalogsRequest,
    settings: Settings = Depends(get_settings),
) -> SmartAnalogsResponse:
    """Find smart analogs for a product."""
    trace_id = uuid.uuid4().hex if settings.enable_request_tracing else None
    service = get_smart_analogs_service(settings)
    return await service.find_analogs(
        request=request,
        trace_id=trace_id,
    )


@router.post(
    "/course/calculate",
    response_model=CourseCalculatorResult,
    summary="Calculate packages needed for a course",
    description="Calculate how many packages are needed for a treatment course based on dosage and duration.",
)
async def calculate_course(
    request: CourseCalculatorRequest,
    settings: Settings = Depends(get_settings),
) -> CourseCalculatorResult:
    """Calculate course requirements."""
    trace_id = uuid.uuid4().hex if settings.enable_request_tracing else None
    service = get_course_calculator_service(settings)
    return await service.calculate(
        request=request,
        trace_id=trace_id,
    )


@router.post(
    "/personalization/context",
    response_model=PurchaseHistoryResponse,
    summary="Get personalization context based on purchase history",
    description="Get user's purchase history context for personalized chat experience.",
)
async def get_personalization_context(
    request: PurchaseHistoryRequest,
    settings: Settings = Depends(get_settings),
) -> PurchaseHistoryResponse:
    """Get personalization context."""
    trace_id = uuid.uuid4().hex if settings.enable_request_tracing else None
    service = get_purchase_history_service(settings)
    return await service.get_purchase_context(
        request=request,
        trace_id=trace_id,
    )


@router.get("/debug/llm-calls")
async def get_llm_debug_calls(
    limit: int = Query(default=20, ge=1, le=100),
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Get recent LLM calls for debugging.
    
    Returns detailed information about recent LLM calls including:
    - Full prompts sent to LLM
    - Raw responses from LLM
    - Token usage
    - Latency
    
    Only available in debug mode.
    """
    if not settings.debug:
        return {"error": "Debug mode is disabled", "records": []}
    
    store = get_llm_debug_store()
    records = store.get_records(limit=limit)
    return {
        "total": len(records),
        "records": records,
    }


@router.delete("/debug/llm-calls")
async def clear_llm_debug_calls(
    settings: Settings = Depends(get_settings),
) -> dict:
    """Clear LLM debug call history."""
    if not settings.debug:
        return {"error": "Debug mode is disabled"}
    
    store = get_llm_debug_store()
    store.clear()
    return {"status": "cleared"}

