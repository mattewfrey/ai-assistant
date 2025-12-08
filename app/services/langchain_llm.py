from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from ..config import Settings
from ..models.assistant import AssistantResponse, Reply
from ..prompts.beautify_prompt import build_beautify_prompt
from ..prompts.intent_prompt import build_intent_prompt
from .cache import CachingService, get_caching_service
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
    ) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for LangChain mode")

        if llm is None:
            llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=settings.openai_temperature,
                timeout=settings.http_timeout_seconds,
                base_url=settings.openai_base_url,
            )
        self._llm = llm
        self._cache = cache or get_caching_service()
        schema_hint = json.dumps(AssistantResponse.model_json_schema(), ensure_ascii=False, indent=2)
        self._intent_prompt = build_intent_prompt(schema_hint)
        self._beautify_prompt = build_beautify_prompt()
        self._intent_llm = self._llm.bind(response_format={"type": "json_object"})
        self._beautify_llm = self._llm.bind(response_format={"type": "json_object"})

    async def parse_intent(
        self,
        *,
        message: str,
        profile: Dict[str, Any] | None,
        dialog_state: DialogState | None,
        ui_state: Dict[str, Any] | None,
        available_intents: list[str],
    ) -> LLMRunResult:
        normalized_message = message.strip().lower()
        profile_signature = json.dumps(profile or {}, ensure_ascii=False, sort_keys=True)
        cached = self._cache.get_llm_response(normalized_message, profile_signature)
        if cached:
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
        prompt_messages = self._intent_prompt.format_prompt(**payload).to_messages()
        try:
            ai_message = await self._intent_llm.ainvoke(prompt_messages)
            response = self._parse_assistant_response(ai_message)
            self._cache.set_llm_response(
                normalized_message,
                profile_signature,
                response.model_dump(),
                ttl_seconds=900,
            )
            usage = self._extract_usage(ai_message)
            return LLMRunResult(response=response, token_usage=usage, cached=False)
        except Exception as exc:  # pragma: no cover - network level failures
            logger.exception("LangChain parse_intent failed: %s", exc)
            fallback = self._fallback_assistant_response(message=message)
            return LLMRunResult(response=fallback, token_usage={}, cached=False)

    async def beautify_reply(
        self,
        *,
        base_reply: Reply,
        data: Dict[str, Any],
        constraints: Dict[str, Any] | None = None,
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
                return BeautifyResult(reply=Reply.model_validate(cached), cached=True)
            except Exception:
                pass  # Cache miss on validation failure
        
        payload = {
            "base_reply": json.dumps(base_reply.model_dump(), ensure_ascii=False),
            "data_json": json.dumps(data, ensure_ascii=False),
            "constraints_json": json.dumps(constraints or {}, ensure_ascii=False),
        }
        prompt_messages = self._beautify_prompt.format_prompt(**payload).to_messages()
        try:
            ai_message = await self._beautify_llm.ainvoke(prompt_messages)
            result = self._parse_reply_json(ai_message)
            # Cache the result
            self._cache.set_beautify_response(
                base_reply.text,
                data_hash,
                constraints_hash,
                result.model_dump(),
                ttl_seconds=600,
            )
            return BeautifyResult(reply=result, cached=False)
        except Exception as exc:  # pragma: no cover - network level failures
            logger.exception("LangChain beautify_reply failed: %s", exc)
            return BeautifyResult(reply=base_reply, cached=False)

    @staticmethod
    def _parse_assistant_response(message: AIMessage) -> AssistantResponse:
        content = _extract_message_content(message)
        try:
            parsed = json.loads(content)
            return AssistantResponse.model_validate(parsed)
        except Exception as exc:  # pragma: no cover - parser fallback
            logger.warning("Failed to parse AssistantResponse JSON: %s; payload=%s", exc, content)
            return LangchainLLMClient._fallback_assistant_response()

    @staticmethod
    def _parse_reply_json(message: AIMessage) -> Reply:
        content = _extract_message_content(message)
        try:
            parsed = json.loads(content)
            return Reply.model_validate(parsed)
        except Exception as exc:  # pragma: no cover - parser fallback
            logger.warning("Failed to parse Reply JSON: %s; payload=%s", exc, content)
            return Reply(text=content if isinstance(content, str) else "Готово.")

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
