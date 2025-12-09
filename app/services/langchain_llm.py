from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from typing import Any, Dict

from langsmith import Client as LangSmithClient, traceable
from langsmith.callbacks import LangChainCallbackHandler
from langchain_core.caches import InMemoryCache
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import OutputParserException
from langchain_core.runnables import RunnableRetry
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from ..config import Settings
from ..models.assistant import AssistantResponse, Reply
from ..prompts.beautify_prompt import build_beautify_prompt
from ..prompts.intent_prompt import build_intent_prompt
from .cache import CachingService, get_caching_service
from .conversation_store import ConversationStore, get_conversation_store
from .dialog_state_store import DialogState

logger = logging.getLogger(__name__)


@dataclass
class LLMRunResult:
    response: AssistantResponse
    token_usage: Dict[str, int]
    cached: bool = False


@dataclass
class BeautifyResult:
    reply: Reply
    cached: bool = False


class LangchainLLMClient:
    """Thin wrapper around LangChain to keep LLM usage centralized."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm: ChatOpenAI | None = None,
        cache: CachingService | None = None,
        conversation_store: ConversationStore | None = None,
        history_limit: int = 10,
    ) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for LangChain mode")

        self._enable_llm_cache(settings)
        self._langsmith_client = self._setup_langsmith(settings)
        self._conversation_store = conversation_store or get_conversation_store()
        self._history_limit = history_limit
        callbacks = self._build_callbacks()
        if llm is None:
            llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=settings.openai_temperature,
                timeout=settings.http_timeout_seconds,
                base_url=settings.openai_base_url,
                callbacks=callbacks or None,
            )
        self._llm = llm
        self._llm_callbacks = callbacks or []
        self._cache = cache or get_caching_service()
        schema_hint = json.dumps(AssistantResponse.model_json_schema(), ensure_ascii=False, indent=2)
        self._intent_prompt = build_intent_prompt(schema_hint)
        self._beautify_prompt = build_beautify_prompt()
        # Structured output with built-in JSON Schema validation
        self._intent_llm = self._llm.with_structured_output(
            AssistantResponse,
            include_raw=True,
            strict=True,
        )
        self._beautify_llm = self._llm.with_structured_output(
            Reply,
            include_raw=True,
            strict=True,
        )
        # Retry chain to auto-recover from transient parsing/validation failures
        base_intent_chain = RunnableRetry(
            self._intent_prompt | self._intent_llm,
            max_attempts=2,
            retry_if_exception_type=(OutputParserException, ValueError),
        )
        self._beautify_chain = RunnableRetry(
            self._beautify_prompt | self._beautify_llm,
            max_attempts=2,
            retry_if_exception_type=(OutputParserException, ValueError),
        )
        # Runnable with history for intent parsing
        self._intent_with_history = RunnableWithMessageHistory(
            base_intent_chain,
            self._history_factory,
            input_messages_key="context_messages",
            history_messages_key="context_messages",
        )

    @staticmethod
    def _setup_langsmith(self, settings: Settings) -> LangSmithClient | None:
        """Initialize LangSmith tracing when enabled via settings/env."""

        tracing_enabled = bool(settings.langsmith_tracing_v2 or settings.langchain_tracing)
        if tracing_enabled:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        if not (settings.langsmith_api_key and settings.langsmith_tracing_v2):
            return None
        try:
            project = settings.langsmith_project or "smart-pharmacy-assistant"
            os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
            os.environ.setdefault("LANGSMITH_PROJECT", project)
            if settings.langsmith_endpoint:
                os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
                os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.langsmith_endpoint)
            else:
                os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            return LangSmithClient(
                api_key=settings.langsmith_api_key,
                api_url=settings.langsmith_endpoint or None,
                project=project,
            )
        except Exception as exc:  # pragma: no cover - tracing is best-effort
            logger.warning("LangSmith client disabled: %s", exc)
            return None

    @staticmethod
    def _enable_llm_cache(settings: Settings) -> None:
        """Enable LangChain in-memory cache if configured."""
        if not settings.langchain_cache:
            return
        set_llm_cache(InMemoryCache())

    @staticmethod
    def _build_invoke_config(
        conversation_id: str | None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        config: Dict[str, Any] = {}
        if conversation_id:
            config["configurable"] = {"session_id": conversation_id}
        if metadata:
            config["metadata"] = metadata
        return config or None

    def _build_callbacks(self) -> list:
        """Return LangSmith callbacks when tracing is configured."""
        if not self._langsmith_client:
            return []
        try:
            return [LangChainCallbackHandler(client=self._langsmith_client)]
        except Exception as exc:  # pragma: no cover - optional tracing
            logger.warning("Failed to initialize LangSmith callback handler: %s", exc)
            return []

    @traceable(run_type="chain", name="parse_intent")
    async def parse_intent(
        self,
        *,
        message: str,
        profile: Dict[str, Any] | None,
        dialog_state: DialogState | None,
        ui_state: Dict[str, Any] | None,
        available_intents: list[str],
        conversation_id: str | None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> LLMRunResult:
        normalized_message = message.strip().lower()
        profile_signature = json.dumps(profile or {}, ensure_ascii=False, sort_keys=True)
        cached = self._cache.get_llm_response(normalized_message, profile_signature)
        if cached:
            logger.info(
                "trace_id=%s user_id=%s conversation_id=%s intent=parse_intent cached=True",
                trace_id or "-",
                user_id or "-",
                conversation_id or "-",
            )
            return LLMRunResult(
                response=AssistantResponse.model_validate(cached),
                token_usage={},
                cached=True,
            )
        payload = {
            "message": message,
            "profile_json": profile_signature,
            "dialog_state_json": json.dumps(self._dialog_state_to_dict(dialog_state), ensure_ascii=False),
            "ui_state_json": json.dumps(ui_state or {}, ensure_ascii=False),
            "available_intents": ", ".join(sorted(available_intents)),
            "context_messages": [],
        }
        metadata = self._build_intent_metadata(
            conversation_id=conversation_id,
            user_id=user_id,
            dialog_state=dialog_state,
            available_intents=available_intents,
            ui_state=ui_state,
        )
        config = self._build_invoke_config(conversation_id, metadata)
        try:
            if conversation_id:
                result = await self._intent_with_history.ainvoke(
                    payload,
                    config=config,
                )
            else:
                result = await self._invoke_with_retry(
                    self._intent_prompt,
                    self._intent_llm,
                    payload,
                    metadata=metadata,
                    conversation_id=conversation_id,
                )
            response, ai_message = self._parse_assistant_response(result)
            self._cache.set_llm_response(
                normalized_message,
                profile_signature,
                response.model_dump(),
                ttl_seconds=900,
            )
            usage = self._extract_usage(ai_message)
            logger.info(
                "trace_id=%s user_id=%s conversation_id=%s intent=parse_intent cached=False tokens=%s",
                trace_id or "-",
                user_id or "-",
                conversation_id or "-",
                usage,
            )
            return LLMRunResult(response=response, token_usage=usage, cached=False)
        except Exception as exc:  # pragma: no cover - network level failures
            logger.exception("LangChain parse_intent failed: %s", exc)
            fallback = self._fallback_assistant_response(message=message)
            return LLMRunResult(response=fallback, token_usage={}, cached=False)

    @traceable(run_type="chain", name="beautify_reply")
    async def beautify_reply(
        self,
        *,
        base_reply: Reply,
        data: Dict[str, Any],
        constraints: Dict[str, Any] | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
        intent: str | None = None,
        router_matched: bool | None = None,
        slot_filling_used: bool | None = None,
        channel: str | None = None,
        metadata: Dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> BeautifyResult:
        """Polish the reply text using LLM with caching support."""
        # Compute cache keys
        data_hash = self._cache.compute_data_hash(data)
        constraints_hash = self._cache.compute_constraints_hash(constraints)
        
        # Check cache
        cached = self._cache.get_beautify_response(
            base_reply.text,
            data_hash,
            constraints_hash,
        )
        if cached:
            try:
                logger.info(
                    "trace_id=%s user_id=%s conversation_id=%s intent=beautify_reply cached=True",
                    trace_id or "-",
                    user_id or "-",
                    conversation_id or "-",
                )
                return BeautifyResult(reply=Reply.model_validate(cached), cached=True)
            except Exception:
                pass  # Cache miss on validation failure
        
        payload = {
            "base_reply": json.dumps(base_reply.model_dump(), ensure_ascii=False),
            "data_json": json.dumps(data, ensure_ascii=False),
            "constraints_json": json.dumps(constraints or {}, ensure_ascii=False),
        }
        combined_metadata = self._build_beautify_metadata(
            conversation_id=conversation_id,
            user_id=user_id,
            intent=intent,
            router_matched=router_matched,
            slot_filling_used=slot_filling_used,
            channel=channel,
            extra=metadata,
        )
        try:
            result = await self._invoke_with_retry(
                self._beautify_prompt,
                self._beautify_llm,
                payload,
                beautify=True,
                metadata=combined_metadata,
                conversation_id=conversation_id,
            )
            reply, _ = self._parse_reply_json(result)
            # Cache the result
            self._cache.set_beautify_response(
                base_reply.text,
                data_hash,
                constraints_hash,
                reply.model_dump(),
                ttl_seconds=600,
            )
            logger.info(
                "trace_id=%s user_id=%s conversation_id=%s intent=beautify_reply cached=False",
                trace_id or "-",
                user_id or "-",
                conversation_id or "-",
            )
            return BeautifyResult(reply=reply, cached=False)
        except Exception as exc:  # pragma: no cover - network level failures
            logger.exception("LangChain beautify_reply failed: %s", exc)
            return BeautifyResult(reply=base_reply, cached=False)

    async def _invoke_with_retry(
        self,
        prompt,
        llm_runnable,
        payload: Dict[str, Any],
        beautify: bool = False,
        metadata: Dict[str, Any] | None = None,
        conversation_id: str | None = None,
    ):
        """Helper to run prompt+LLM with retry when history is not used."""
        chain = self._beautify_chain if beautify else RunnableRetry(
            prompt | llm_runnable,
            max_attempts=2,
            retry_if_exception_type=(OutputParserException, ValueError),
        )
        config = self._build_invoke_config(conversation_id=conversation_id, metadata=metadata)
        return await chain.ainvoke(payload, config=config)

    def _parse_assistant_response(self, result: Any) -> tuple[AssistantResponse, AIMessage | None]:
        parsed = result
        raw_message: AIMessage | None = None
        if isinstance(result, dict) and "parsed" in result:
            parsed = result.get("parsed")
            raw_message = result.get("raw")
        elif hasattr(result, "raw"):
            raw_message = getattr(result, "raw", None)
            parsed = getattr(result, "parsed", result)
        if isinstance(parsed, AssistantResponse):
            return parsed, raw_message
        if isinstance(parsed, AIMessage):
            raw_message = parsed
            content = _extract_message_content(parsed)
            try:
                return AssistantResponse.model_validate_json(content), raw_message
            except Exception as exc:
                logger.warning("Failed to parse AssistantResponse JSON: %s; payload=%s", exc, content)
                return LangchainLLMClient._fallback_assistant_response(), raw_message
        try:
            return AssistantResponse.model_validate(parsed), raw_message
        except Exception as exc:  # pragma: no cover - parser fallback
            logger.warning("Failed to parse AssistantResponse: %s; payload=%s", exc, parsed)
            return LangchainLLMClient._fallback_assistant_response(), raw_message

    def _parse_reply_json(self, result: Any) -> tuple[Reply, AIMessage | None]:
        parsed = result
        raw_message: AIMessage | None = None
        if isinstance(result, dict) and "parsed" in result:
            parsed = result.get("parsed")
            raw_message = result.get("raw")
        elif hasattr(result, "raw"):
            raw_message = getattr(result, "raw", None)
            parsed = getattr(result, "parsed", result)
        if isinstance(parsed, Reply):
            return parsed, raw_message
        if isinstance(parsed, AIMessage):
            raw_message = parsed
            content = _extract_message_content(parsed)
            try:
                return Reply.model_validate_json(content), raw_message
            except Exception as exc:
                logger.warning("Failed to parse Reply JSON: %s; payload=%s", exc, content)
                return Reply(text=content if isinstance(content, str) else "Готово."), raw_message
        try:
            return Reply.model_validate(parsed), raw_message
        except Exception as exc:  # pragma: no cover - parser fallback
            logger.warning("Failed to parse Reply: %s; payload=%s", exc, parsed)
            return Reply(text=str(parsed) if parsed else "Готово."), raw_message

    def _history_factory(self, session_id: str) -> BaseChatMessageHistory:
        return _ConversationStoreHistory(self._conversation_store, session_id, self._history_limit)

    @staticmethod
    def _extract_usage(message: AIMessage) -> Dict[str, int]:
        metadata = getattr(message, "response_metadata", {}) or {}
        usage = metadata.get("token_usage") or {}
        return {k: int(v) for k, v in usage.items() if isinstance(v, (int, float))}

    @staticmethod
    def _dialog_state_to_dict(dialog_state: DialogState | None) -> Dict[str, Any]:
        if dialog_state is None:
            return {}
        return {
            "intent": getattr(dialog_state.current_intent, "value", dialog_state.current_intent),
            "channel": getattr(dialog_state.channel, "value", dialog_state.channel),
            "slots": dialog_state.slots,
            "context_products": dialog_state.context_products,
            "last_reply": dialog_state.last_reply,
        }

    @staticmethod
    def _build_intent_metadata(
        *,
        conversation_id: str | None,
        user_id: str | None,
        dialog_state: DialogState | None,
        available_intents: list[str],
        ui_state: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        metadata = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "channel": getattr(dialog_state, "channel", None),
            "intent": getattr(dialog_state, "current_intent", None),
            "router_matched": False,
            "slot_filling_used": False,
            "available_intents": sorted(available_intents),
        }
        if ui_state:
            metadata["ui_state"] = {k: v for k, v in ui_state.items() if v is not None}
        return LangchainLLMClient._normalize_metadata(metadata)

    @staticmethod
    def _build_beautify_metadata(
        *,
        conversation_id: str | None,
        user_id: str | None,
        intent: str | None,
        router_matched: bool | None,
        slot_filling_used: bool | None,
        channel: str | None,
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        metadata = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "intent": intent,
            "router_matched": router_matched,
            "slot_filling_used": slot_filling_used,
            "channel": channel,
        }
        if extra:
            metadata.update(extra)
        return LangchainLLMClient._normalize_metadata(metadata)

    @staticmethod
    def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if hasattr(value, "value"):
                normalized[key] = getattr(value, "value")
            else:
                normalized[key] = value
        return normalized

    @staticmethod
    def _fallback_assistant_response(message: str | None = None) -> AssistantResponse:
        reply_text = (
            "Не удалось обработать запрос автоматически. "
            "Пожалуйста, переформулируйте вопрос или выберите подсказку ниже."
        )
        if message:
            reply_text += f" Последнее сообщение: \"{message.strip()}\"."
        return AssistantResponse(
            reply=Reply(text=reply_text),
            actions=[],
            meta=None,
        )


def _extract_message_content(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        # OpenAI can return list[dict]; join textual segments
        return " ".join(
            part.get("text", "")
            for part in message.content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(message.content)


class _ConversationStoreHistory(BaseChatMessageHistory):
    """Adapter to expose ConversationStore as LangChain message history."""

    def __init__(self, store: ConversationStore, session_id: str, limit: int = 10) -> None:
        self._store = store
        self._session_id = session_id or ""
        self._limit = limit

    @property
    def messages(self) -> list[BaseMessage]:
        history = self._store.get_history(self._session_id, limit=self._limit)
        result: list[BaseMessage] = []
        for item in history:
            if item.role == "assistant":
                result.append(AIMessage(content=item.content))
            else:
                result.append(HumanMessage(content=item.content))
        return result

    def add_message(self, message: BaseMessage) -> None:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        self._store.append_message(self._session_id, role, str(message.content))

    def clear(self) -> None:
        # No-op; clearing is managed outside this adapter.
        return
