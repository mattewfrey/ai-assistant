"""
YandexGPT integration via REST API.

Прямая интеграция с YandexGPT без SDK — только HTTP запросы.
Реализует интерфейс LangChain BaseChatModel для совместимости.

REST API docs: https://yandex.cloud/docs/foundation-models/text-generation/api-ref/
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

logger = logging.getLogger(__name__)

# YandexGPT REST API endpoint
YANDEX_GPT_ENDPOINT = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


class ChatYandexGPTDirect(BaseChatModel):
    """
    LangChain-compatible chat model for YandexGPT via REST API.
    
    Не требует yandexcloud SDK — работает через обычные HTTP запросы.
    
    Usage:
        llm = ChatYandexGPTDirect(
            api_key="your-api-key",
            folder_id="your-folder-id",
            model="yandexgpt/latest",
            temperature=0.3,
        )
        response = llm.invoke([HumanMessage(content="Привет!")])
    """
    
    api_key: str = Field(..., description="Yandex Cloud API key")
    folder_id: str = Field(..., description="Yandex Cloud folder ID")
    model: str = Field(default="yandexgpt/latest", description="Model name")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2000, description="Max tokens in response")
    timeout: float = Field(default=30.0, description="HTTP timeout in seconds")
    
    @property
    def _llm_type(self) -> str:
        return "yandexgpt-direct"
    
    @property
    def _model_uri(self) -> str:
        """Build model URI for YandexGPT API."""
        return f"gpt://{self.folder_id}/{self.model}"
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to YandexGPT format."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"
            
            result.append({
                "role": role,
                "text": msg.content if isinstance(msg.content, str) else str(msg.content),
            })
        return result
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from YandexGPT."""
        
        payload = {
            "modelUri": self._model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": str(self.max_tokens),
            },
            "messages": self._convert_messages(messages),
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id,
        }
        
        logger.debug(
            "YandexGPT request: model=%s, messages=%d, temperature=%.2f",
            self.model,
            len(messages),
            self.temperature,
        )
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    YANDEX_GPT_ENDPOINT,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error("YandexGPT HTTP error: %s - %s", e.response.status_code, error_body)
            raise ValueError(f"YandexGPT API error: {e.response.status_code} - {error_body}") from e
        except httpx.RequestError as e:
            logger.error("YandexGPT connection error: %s", str(e))
            raise ValueError(f"YandexGPT connection error: {str(e)}") from e
        
        # Parse response
        try:
            result = data.get("result", {})
            alternatives = result.get("alternatives", [])
            
            if not alternatives:
                raise ValueError("YandexGPT returned empty response")
            
            # Get first alternative
            alt = alternatives[0]
            message_data = alt.get("message", {})
            text = message_data.get("text", "")
            
            # Token usage
            usage = result.get("usage", {})
            input_tokens = int(usage.get("inputTextTokens", 0))
            output_tokens = int(usage.get("completionTokens", 0))
            
            total_tokens = input_tokens + output_tokens
            logger.info(
                "YandexGPT usage: input=%d, output=%d, total=%d tokens",
                input_tokens,
                output_tokens,
                total_tokens,
            )
            
        except (KeyError, TypeError, IndexError) as e:
            logger.error("YandexGPT parse error: %s, response: %s", str(e), data)
            raise ValueError(f"Failed to parse YandexGPT response: {e}") from e
        
        # Add token usage to response_metadata for compatibility with LangChain extraction
        ai_message = AIMessage(
            content=text,
            response_metadata={
                "token_usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
                "model_name": self.model,
            },
        )
        generation = ChatGeneration(message=ai_message)
        
        return ChatResult(
            generations=[generation],
            llm_output={
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        )
    
    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters for caching."""
        return {
            "model": self.model,
            "folder_id": self.folder_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# Async version for use with async code
class AsyncChatYandexGPTDirect(ChatYandexGPTDirect):
    """Async version of ChatYandexGPTDirect."""
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate response from YandexGPT."""
        
        payload = {
            "modelUri": self._model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": str(self.max_tokens),
            },
            "messages": self._convert_messages(messages),
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id,
        }
        
        logger.debug(
            "YandexGPT async request: model=%s, messages=%d",
            self.model,
            len(messages),
        )
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    YANDEX_GPT_ENDPOINT,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error("YandexGPT HTTP error: %s - %s", e.response.status_code, error_body)
            raise ValueError(f"YandexGPT API error: {e.response.status_code} - {error_body}") from e
        except httpx.RequestError as e:
            logger.error("YandexGPT connection error: %s", str(e))
            raise ValueError(f"YandexGPT connection error: {str(e)}") from e
        
        # Parse response (same as sync)
        try:
            result = data.get("result", {})
            alternatives = result.get("alternatives", [])
            
            if not alternatives:
                raise ValueError("YandexGPT returned empty response")
            
            alt = alternatives[0]
            message_data = alt.get("message", {})
            text = message_data.get("text", "")
            
            usage = result.get("usage", {})
            input_tokens = int(usage.get("inputTextTokens", 0))
            output_tokens = int(usage.get("completionTokens", 0))
            
        except (KeyError, TypeError, IndexError) as e:
            logger.error("YandexGPT parse error: %s", str(e))
            raise ValueError(f"Failed to parse YandexGPT response: {e}") from e
        
        # Add token usage to response_metadata for compatibility with LangChain extraction
        total_tokens = input_tokens + output_tokens
        ai_message = AIMessage(
            content=text,
            response_metadata={
                "token_usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
                "model_name": self.model,
            },
        )
        generation = ChatGeneration(message=ai_message)
        
        return ChatResult(
            generations=[generation],
            llm_output={
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        )
