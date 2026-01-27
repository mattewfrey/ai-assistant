"""
Gradio Demo for Product AI Chat service (Gradio 6.x compatible).

Run with:
    python demo/gradio_app.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from collections import deque
from datetime import datetime
from typing import Any

import gradio as gr
import httpx

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API base URL
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_TIMEOUT = 30.0


# =============================================================================
# Helpers
# =============================================================================
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_gradio_auth(value: str | None) -> tuple[str, str] | list[tuple[str, str]] | None:
    """
    Parse GRADIO_AUTH env var.

    Formats:
      - "user:pass"
      - "user:pass,user2:pass2"
    """
    if not value or not value.strip():
        return None
    pairs: list[tuple[str, str]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            continue
        user, pwd = item.split(":", 1)
        user = user.strip()
        pwd = pwd.strip()
        if user and pwd:
            pairs.append((user, pwd))
    if not pairs:
        return None
    return pairs[0] if len(pairs) == 1 else pairs


# =============================================================================
# Log Collector - captures logs from the backend
# =============================================================================
class LogCollector:
    """Collects logs from API responses and stores them for display."""
    
    def __init__(self, max_entries: int = 500):
        self._entries: deque[dict] = deque(maxlen=max_entries)
        self._lock = threading.Lock()
    
    def add_entry(
        self,
        level: str,
        source: str,
        message: str,
        trace_id: str | None = None,
        extra: dict | None = None,
    ) -> None:
        """Add a log entry."""
        with self._lock:
            self._entries.append({
                "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "level": level,
                "source": source,
                "message": message,
                "trace_id": trace_id or "-",
                "extra": extra or {},
            })
    
    def add_request(self, endpoint: str, payload: dict | None = None) -> None:
        """Log an outgoing request."""
        # Extract message for display
        message_preview = ""
        if payload:
            msg = payload.get("message", "")
            if msg:
                message_preview = f" | msg: \"{msg[:50]}{'...' if len(msg) > 50 else ''}\""
            product_id = payload.get("product_id", "")
            if product_id:
                message_preview += f" | product: {product_id[:20]}..."
        
        self.add_entry(
            level="REQUEST",
            source="gradio",
            message=f"→ {endpoint}{message_preview}",
            extra={"payload": payload},
        )
    
    def add_response(self, endpoint: str, response: dict, trace_id: str | None = None) -> None:
        """Log a response and extract debug info."""
        # Extract debug info from response
        meta = response.get("meta", {})
        debug = meta.get("debug", {})
        
        # Extract reply text preview
        reply = response.get("reply", {})
        reply_text = reply.get("text", "") if isinstance(reply, dict) else ""
        reply_preview = f" | reply: \"{reply_text[:60]}{'...' if len(reply_text) > 60 else ''}\"" if reply_text else ""
        
        # Confidence
        confidence = meta.get("confidence", debug.get("confidence", 0))
        conf_str = f" | conf: {confidence:.0%}" if confidence else ""
        
        # Log main response info
        self.add_entry(
            level="RESPONSE",
            source="api",
            message=f"← {endpoint}{conf_str}{reply_preview}",
            trace_id=trace_id or debug.get("trace_id"),
            extra={"debug": debug},
        )
        
        # Extract and log pipeline details
        if debug:
            self._log_debug_details(debug)
    
    def _log_debug_details(self, debug: dict) -> None:
        """Extract meaningful info from debug payload."""
        trace_id = debug.get("trace_id", "-")
        
        # Detect Product Chat vs Main Chat by checking for product_id in debug
        is_product_chat = "product_id" in debug or "context_hash" in debug
        
        if is_product_chat:
            self._log_product_chat_debug(debug, trace_id)
        else:
            self._log_main_chat_debug(debug, trace_id)
    
    def _log_product_chat_debug(self, debug: dict, trace_id: str) -> None:
        """Log Product Chat specific debug info."""
        product_id = debug.get("product_id", "-")
        llm_used = debug.get("llm_used", False)
        llm_cached = debug.get("llm_cached", False)
        model = debug.get("model", "unknown")
        
        # Context info
        context_cache_hit = debug.get("context_cache_hit", False)
        context_hash = debug.get("context_hash", "")[:12] if debug.get("context_hash") else "-"
        
        self.add_entry(
            level="CONTEXT",
            source="product_gateway",
            message=f"Product: {product_id[:20]}... | Context hash: {context_hash} | Cache: {'✓' if context_cache_hit else '✗'}",
            trace_id=trace_id,
        )
        
        # Policy check
        out_of_scope = debug.get("out_of_scope", False)
        injection_detected = debug.get("injection_detected", False)
        refusal_reason = debug.get("refusal_reason")
        
        if out_of_scope or injection_detected or refusal_reason:
            self.add_entry(
                level="POLICY",
                source="policy_guard",
                message=f"Out of scope: {'✓' if out_of_scope else '✗'} | Injection: {'✓' if injection_detected else '✗'} | Refusal: {refusal_reason or 'none'}",
                trace_id=trace_id,
            )
        
        # LLM details
        if llm_used:
            token_usage = debug.get("token_usage", {})
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            
            self.add_entry(
                level="LLM",
                source="llm",
                message=f"Model: {model} | Cached: {'✓' if llm_cached else '✗'} | Tokens: {prompt_tokens}→{completion_tokens}",
                trace_id=trace_id,
            )
        
        # Used fields (citations)
        used_fields = debug.get("used_fields", [])
        if used_fields:
            self.add_entry(
                level="CITATIONS",
                source="llm",
                message=f"Used fields: {', '.join(used_fields[:10])}" + (f" (+{len(used_fields)-10} more)" if len(used_fields) > 10 else ""),
                trace_id=trace_id,
            )
    
    def _log_main_chat_debug(self, debug: dict, trace_id: str) -> None:
        """Log Main Chat specific debug info."""
        # Pipeline path
        pipeline = debug.get("pipeline_path", "unknown")
        router_matched = debug.get("router_matched", False)
        llm_used = debug.get("llm_used", False)
        
        self.add_entry(
            level="PIPELINE",
            source="backend",
            message=f"Pipeline: {pipeline} | Router: {'✓' if router_matched else '✗'} | LLM: {'✓' if llm_used else '✗'}",
            trace_id=trace_id,
        )
        
        # Intent chain
        intent_chain = debug.get("intent_chain", [])
        if intent_chain:
            self.add_entry(
                level="INTENT",
                source="nlu",
                message=f"Intents: {' → '.join(intent_chain)}",
                trace_id=trace_id,
            )
        
        # Router details
        if router_matched:
            confidence = debug.get("router_confidence", 0)
            match_type = debug.get("router_match_type", "unknown")
            triggers = debug.get("matched_triggers", [])
            self.add_entry(
                level="ROUTER",
                source="router",
                message=f"Match: {match_type} | Confidence: {confidence:.2f} | Triggers: {triggers[:5]}",
                trace_id=trace_id,
            )
        
        # LLM details
        if llm_used:
            llm_confidence = debug.get("llm_confidence", 0)
            llm_cached = debug.get("llm_cached", False)
            reasoning = debug.get("llm_reasoning", "")
            self.add_entry(
                level="LLM",
                source="llm",
                message=f"Confidence: {llm_confidence:.2f} | Cached: {'✓' if llm_cached else '✗'}" + 
                        (f" | Reasoning: {reasoning[:100]}..." if reasoning else ""),
                trace_id=trace_id,
            )
        
        # Extracted entities
        entities = debug.get("extracted_entities", {})
        if entities:
            self.add_entry(
                level="SLOTS",
                source="extraction",
                message=f"Extracted: {entities}",
                trace_id=trace_id,
            )
        
        # Slot filling
        if debug.get("slot_filling_used"):
            missing = debug.get("missing_slots", [])
            filled = debug.get("filled_slots", [])
            self.add_entry(
                level="SLOTS",
                source="slot_manager",
                message=f"Filled: {filled} | Missing: {missing}",
                trace_id=trace_id,
            )
    
    def get_entries(self, limit: int = 100) -> list[dict]:
        """Get recent log entries."""
        with self._lock:
            entries = list(self._entries)
            return entries[-limit:]
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()
    
    def format_as_text(self, limit: int = 100, filter_level: str | None = None) -> str:
        """Format entries as readable text."""
        entries = self.get_entries(limit)
        
        if filter_level and filter_level != "ALL":
            entries = [e for e in entries if e["level"] == filter_level]
        
        if not entries:
            return "📭 Логов пока нет. Отправьте запрос в любую вкладку."
        
        lines = []
        for entry in entries:
            level_emoji = {
                "REQUEST": "📤",
                "RESPONSE": "📥",
                "PIPELINE": "🔀",
                "INTENT": "🎯",
                "CONTEXT": "📋",
                "POLICY": "🛡️",
                "CITATIONS": "📎",
                "ROUTER": "🔍",
                "LLM": "🤖",
                "SLOTS": "📦",
                "ERROR": "❌",
                "INFO": "ℹ️",
            }.get(entry["level"], "•")
            
            line = f"`{entry['timestamp']}` {level_emoji} **{entry['level']}** [{entry['source']}]"
            if entry["trace_id"] != "-":
                line += f" `{entry['trace_id'][:8]}`"
            line += f"\n{entry['message']}"
            
            lines.append(line)
        
        return "\n\n---\n\n".join(lines)
    
    def format_as_json(self, limit: int = 50) -> str:
        """Format entries as JSON for debugging."""
        entries = self.get_entries(limit)
        return json.dumps(entries, ensure_ascii=False, indent=2)


# Global log collector instance
log_collector = LogCollector()


def api_call(method: str, endpoint: str, json_data: dict | None = None) -> dict:
    """Make API call to our backend with logging."""
    url = f"{API_BASE}{endpoint}"
    
    # Log the request
    log_collector.add_request(endpoint, json_data)
    
    try:
        with httpx.Client(timeout=API_TIMEOUT) as client:
            if method == "GET":
                resp = client.get(url)
            elif method == "DELETE":
                resp = client.delete(url)
            else:
                resp = client.post(url, json=json_data)
            resp.raise_for_status()
            result = resp.json()
            
            # Log the response with debug info extraction
            trace_id = result.get("meta", {}).get("debug", {}).get("trace_id")
            log_collector.add_response(endpoint, result, trace_id)
            
            return result
    except httpx.HTTPStatusError as e:
        error_result = {"error": f"HTTP {e.response.status_code}", "detail": e.response.text}
        log_collector.add_entry(
            level="ERROR",
            source="api",
            message=f"HTTP Error {e.response.status_code}: {e.response.text[:200]}",
        )
        return error_result
    except httpx.RequestError as e:
        error_result = {"error": "Connection Error", "detail": str(e)}
        log_collector.add_entry(
            level="ERROR",
            source="api",
            message=f"Connection Error: {str(e)}",
        )
        return error_result


# =============================================================================
# Tab 0: Main Assistant Chat
# =============================================================================
def chat_with_assistant(
    message: str,
    history: list[dict],
    user_id: str,
    conversation_id: str,
) -> tuple[list[dict], str, str]:
    """Send message to Main AI Assistant. Returns updated history, conversation_id, and debug info."""
    if not message.strip():
        return history, conversation_id, ""

    payload = {
        "message": message.strip(),
        "user_id": user_id.strip() if user_id.strip() else "demo-user",
    }
    if conversation_id.strip():
        payload["conversation_id"] = conversation_id.strip()

    result = api_call("POST", "/api/ai/chat/message", payload)

    if "error" in result:
        bot_msg = f"❌ {result['error']}: {result.get('detail', '')}"
        debug_info = ""
    else:
        reply = result.get("reply", {})
        text = reply.get("text", "Нет ответа")
        
        # Format response with additional info
        bot_msg = text
        
        # Add product cards if available
        data = result.get("data", {})
        products = data.get("products", [])
        if products:
            bot_msg += f"\n\n📦 Найдено товаров: {len(products)}"
            for p in products[:3]:
                name = p.get("name", p.get("title", "Товар"))
                price = p.get("price", "")
                bot_msg += f"\n• {name}" + (f" — {price}₽" if price else "")
            if len(products) > 3:
                bot_msg += f"\n• ... и ещё {len(products) - 3}"
        
        # Add quick replies if available
        meta = result.get("meta", {})
        quick_replies = meta.get("quick_replies", [])
        if quick_replies:
            bot_msg += "\n\n💡 " + " | ".join(quick_replies[:4])
        
        # Update conversation_id from response
        conversation_id = result.get("conversation_id", conversation_id)
        
        # Format debug info
        debug = meta.get("debug", {})
        debug_info = _format_debug_summary(debug)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": bot_msg},
    ]
    return history, conversation_id, debug_info


def _format_debug_summary(debug: dict) -> str:
    """Format debug info as a compact summary."""
    if not debug:
        return ""
    
    lines = []
    
    # Pipeline
    pipeline = debug.get("pipeline_path", "unknown")
    lines.append(f"**Pipeline:** `{pipeline}`")
    
    # Intent
    intents = debug.get("intent_chain", [])
    if intents:
        lines.append(f"**Intent:** `{intents[-1]}`")
    
    # Router
    if debug.get("router_matched"):
        conf = debug.get("router_confidence", 0)
        match_type = debug.get("router_match_type", "")
        lines.append(f"**Router:** ✓ {match_type} ({conf:.0%})")
    
    # LLM
    if debug.get("llm_used"):
        cached = "cached" if debug.get("llm_cached") else "fresh"
        llm_conf = debug.get("llm_confidence", 0)
        lines.append(f"**LLM:** ✓ {cached} ({llm_conf:.0%})")
    
    # Slots
    entities = debug.get("extracted_entities", {})
    if entities:
        slots_str = ", ".join(f"{k}={v}" for k, v in list(entities.items())[:3])
        lines.append(f"**Slots:** {slots_str}")
    
    # Trace ID
    trace_id = debug.get("trace_id", "")
    if trace_id:
        lines.append(f"**Trace:** `{trace_id[:12]}...`")
    
    return "\n".join(lines)


# =============================================================================
# Tab 1: Product Chat (Gradio 6.x format)
# =============================================================================
def init_product_chat(product_id: str, store_id: str) -> tuple[list[dict], str]:
    """Initialize product chat with greeting and AI summary."""
    payload = {
        "product_id": product_id.strip(),
        "store_id": store_id.strip() if store_id.strip() else None,
        "shipping_method": "PICKUP",
    }
    
    result = api_call("POST", "/api/product-ai/chat/init", payload)
    
    if "error" in result:
        return [], ""
    
    conversation_id = result.get("conversation_id", "")
    greeting = result.get("greeting", "")
    ai_summary = result.get("ai_summary")
    
    # Build greeting message (without suggested questions - we have buttons for that)
    greeting_parts = [greeting]
    
    if ai_summary:
        greeting_parts.append(f"\n\n📝 **AI-обзор товара:**\n{ai_summary}")
    
    greeting_parts.append("\n\n👆 *Используйте кнопки быстрых вопросов или введите свой вопрос*")
    
    greeting_message = "".join(greeting_parts)
    
    history = [{"role": "assistant", "content": greeting_message}]
    return history, conversation_id


def chat_with_product(
    message: str,
    history: list[dict],
    product_id: str,
    store_id: str,
    conversation_id: str,
) -> tuple[list[dict], str]:
    """Send message to Product AI Chat. Returns updated history and conversation_id."""
    if not product_id.strip():
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "❌ Ошибка: укажите Product ID"},
        ]
        return history, conversation_id

    if not message.strip():
        return history, conversation_id

    # Initialize chat on first message (empty conversation_id)
    if not conversation_id.strip():
        init_history, conversation_id = init_product_chat(product_id, store_id)
        if init_history:
            history = init_history

    payload = {
        "product_id": product_id.strip(),
        "message": message.strip(),
        "store_id": store_id.strip() if store_id.strip() else None,
        "shipping_method": "PICKUP",
        "conversation_id": conversation_id,
    }

    result = api_call("POST", "/api/product-ai/chat/message", payload)

    if "error" in result:
        bot_msg = f"❌ {result['error']}: {result.get('detail', '')}"
    else:
        reply = result.get("reply", {})
        text = reply.get("text", "Нет ответа")
        citations = result.get("citations", [])
        
        bot_msg = text
        if citations:
            # citations is a list of dicts with field_path
            citation_fields = [c.get("field_path", str(c)) for c in citations]
            bot_msg += f"\n\n📎 Источники: {', '.join(citation_fields)}"
        
        # Update conversation_id from response
        conversation_id = result.get("conversation_id", conversation_id)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": bot_msg},
    ]
    return history, conversation_id


# =============================================================================
# Tab 2: FAQ Generator
# =============================================================================
def generate_faq(product_id: str, force_refresh: bool) -> str:
    """Generate FAQ for a product."""
    if not product_id.strip():
        return "❌ Укажите Product ID"

    endpoint = f"/api/product-ai/faq/{product_id.strip()}"
    if force_refresh:
        endpoint += "?force_refresh=true"

    result = api_call("GET", endpoint)

    if "error" in result:
        return f"❌ {result['error']}: {result.get('detail', '')}"

    output = f"## FAQ для товара: {result.get('product_name', product_id)}\n\n"
    
    for i, faq in enumerate(result.get("faqs", []), 1):
        output += f"### {i}. {faq['question']}\n"
        output += f"{faq['answer']}\n"
        if faq.get("used_fields"):
            output += f"*Источники: {', '.join(faq['used_fields'])}*\n"
        output += "\n"

    meta = result.get("meta", {})
    output += f"\n---\n*Источник: {meta.get('source', 'unknown')} | "
    output += f"Кэш: {'да' if result.get('cache_hit') else 'нет'}*"

    return output


# =============================================================================
# Tab 3: Drug Interactions
# =============================================================================
def check_drug_interactions(drugs_text: str) -> str:
    """Check drug interactions."""
    drugs = [d.strip() for d in drugs_text.split(",") if d.strip()]
    
    if len(drugs) < 2:
        return "❌ Введите минимум 2 препарата через запятую"

    result = api_call("POST", "/api/product-ai/drug-interactions/check", {
        "drugs": drugs
    })

    if "error" in result:
        return f"❌ {result['error']}: {result.get('detail', '')}"

    if not result.get("has_interactions"):
        return f"✅ Взаимодействий между препаратами не обнаружено\n\nПроверено: {', '.join(drugs)}"

    output = "## ⚠️ Обнаружены взаимодействия\n\n"
    
    for interaction in result.get("interactions", []):
        severity = interaction.get("severity", "unknown")
        emoji = {"critical": "🔴", "major": "🟠", "moderate": "🟡", "minor": "🟢"}.get(severity, "⚪")
        
        output += f"### {emoji} {interaction['drug_a']} + {interaction['drug_b']}\n"
        output += f"**Уровень:** {severity}\n\n"
        output += f"{interaction.get('description', '')}\n\n"
        if interaction.get("recommendation"):
            output += f"**Рекомендация:** {interaction['recommendation']}\n\n"
        output += "---\n\n"

    return output


# =============================================================================
# Tab 4: Smart Analogs
# =============================================================================
def find_analogs(
    product_id: str,
    inn: str,
    current_price: float,
    limit: int,
    only_cheaper: bool,
) -> str:
    """Find product analogs by INN."""
    if not inn.strip():
        return "❌ Укажите МНН (действующее вещество)"

    payload = {
        "product_id": product_id.strip() or "unknown",
        "inn": inn.strip(),
        "current_price": current_price if current_price > 0 else None,
        "limit": int(limit),
        "only_cheaper": only_cheaper,
    }

    result = api_call("POST", "/api/product-ai/analogs/find", payload)

    if "error" in result:
        return f"❌ {result['error']}: {result.get('detail', '')}"

    analogs = result.get("analogs", [])
    if not analogs:
        return f"Аналогов с МНН '{inn}' не найдено"

    output = f"## Аналоги по МНН: {result.get('inn', inn)}\n\n"
    output += f"Найдено: {result.get('total_found', len(analogs))}\n\n"

    for i, analog in enumerate(analogs, 1):
        saving = analog.get("savings_amount")
        saving_text = f" (экономия {saving:.0f}₽)" if saving and saving > 0 else ""
        
        output += f"### {i}. {analog['name']}\n"
        output += f"- **Цена:** {analog['price']:.0f}₽{saving_text}\n"
        output += f"- **Производитель:** {analog.get('manufacturer', 'н/д')}\n"
        output += f"- **Форма:** {analog.get('form', 'н/д')}\n"
        if analog.get("in_stock"):
            output += "- ✅ В наличии\n"
        output += "\n"

    return output


# =============================================================================
# Tab 5: Course Calculator
# =============================================================================
def calculate_course(
    product_id: str,
    units_per_package: int,
    dose_per_intake: int,
    frequency: str,
    course_days: int,
    reserve_percent: int,
    price_per_package: float,
) -> str:
    """Calculate medication course requirements."""
    if units_per_package <= 0:
        return "❌ Укажите количество единиц в упаковке"

    payload = {
        "product_id": product_id.strip() or "product",
        "units_per_package": int(units_per_package),
        "dose_per_intake": int(dose_per_intake),
        "frequency": frequency,
        "course_days": int(course_days),
        "add_reserve_percent": int(reserve_percent),
    }

    result = api_call("POST", "/api/product-ai/course/calculate", payload)

    if "error" in result:
        return f"❌ {result['error']}: {result.get('detail', '')}"

    output = "## 📊 Расчёт курса\n\n"
    output += f"### Входные данные\n"
    output += f"- Единиц в упаковке: **{result['units_per_package']}**\n"
    output += f"- Доза за приём: **{result['dose_per_intake']}** ед.\n"
    output += f"- Частота: **{result['intakes_per_day']}** раз/день\n"
    output += f"- Длительность курса: **{result['course_days']}** дней\n\n"

    output += f"### Результат\n"
    output += f"- Всего нужно единиц: **{result['total_units_needed']}**\n"
    output += f"- Упаковок нужно: **{result['packages_needed']}** шт.\n"
    output += f"- Останется единиц: **{result['units_remaining']}**\n"

    if result.get("packages_with_reserve"):
        output += f"\n### С запасом ({result['reserve_percent']}%)\n"
        output += f"- Упаковок с запасом: **{result['packages_with_reserve']}** шт.\n"

    if price_per_package > 0:
        total = result['packages_needed'] * price_per_package
        output += f"\n### Стоимость\n"
        output += f"- Цена за упаковку: {price_per_package:.0f}₽\n"
        output += f"- **Итого: {total:.0f}₽**\n"

    output += f"\n---\n*{result.get('recommendation', '')}*"

    return output


# =============================================================================
# Tab 6: Personalization
# =============================================================================
def get_personalization(user_id: str, product_id: str) -> str:
    """Get personalization context for user."""
    if not user_id.strip():
        return "❌ Укажите User ID"

    payload = {
        "user_id": user_id.strip(),
        "product_id": product_id.strip() or None,
    }

    result = api_call("POST", "/api/product-ai/personalization/context", payload)

    if "error" in result:
        return f"❌ {result['error']}: {result.get('detail', '')}"

    ctx = result.get("context", {})
    profile = result.get("profile", {})

    output = f"## 👤 Персонализация для {user_id}\n\n"
    
    output += "### Профиль пользователя\n"
    output += f"- Статус: **{'Возвращающийся' if ctx.get('is_returning_user') else 'Новый'}** покупатель\n"
    output += f"- Всего покупок: **{profile.get('total_purchases', 0)}**\n"
    output += f"- Общая сумма: **{profile.get('total_spent', 0):.0f}₽**\n"
    
    if profile.get("favorite_categories"):
        output += f"- Любимые категории: {', '.join(profile['favorite_categories'])}\n"

    if ctx.get("personalized_greeting"):
        output += f"\n### Приветствие\n> {ctx['personalized_greeting']}\n"

    if ctx.get("suggested_quantity"):
        output += f"\n### Рекомендации\n"
        output += f"- Рекомендуемое количество: **{ctx['suggested_quantity']}** шт.\n"

    if result.get("also_bought"):
        output += f"\n### С этим товаром покупали\n"
        for item in result["also_bought"][:5]:
            output += f"- {item.get('name', item.get('product_id'))}\n"

    return output


# =============================================================================
# Tab 7: Proactive Hints
# =============================================================================
def get_proactive_hints(product_id: str, trigger_type: str, limit: int) -> str:
    """Get proactive hints for product."""
    if not product_id.strip():
        return "❌ Укажите Product ID"

    payload = {
        "product_id": product_id.strip(),
        "trigger_type": trigger_type,
        "limit": int(limit),
    }

    result = api_call("POST", "/api/product-ai/proactive/hints", payload)

    if "error" in result:
        return f"❌ {result['error']}: {result.get('detail', '')}"

    hints = result.get("hints", [])
    if not hints:
        return "Подсказок для данного триггера нет"

    output = f"## 💡 Проактивные подсказки\n\n"
    output += f"**Триггер:** {trigger_type}\n\n"

    for hint in hints:
        priority = hint.get("priority", 0)
        emoji = "🔥" if priority >= 9 else "⭐" if priority >= 7 else "💬"
        
        output += f"### {emoji} {hint.get('hint_type', 'hint')}\n"
        output += f"{hint.get('message', '')}\n"
        
        if hint.get("suggested_question"):
            output += f"\n*Предложить вопрос:* \"{hint['suggested_question']}\"\n"
        
        output += f"\n*Приоритет: {priority}/10*\n\n"
        output += "---\n\n"

    return output


# =============================================================================
# Custom Theme & CSS for Professional Look
# =============================================================================
CUSTOM_CSS = """
/* Calm, neutral color scheme */
:root {
    --primary-color: #475569;
    --primary-hover: #334155;
    --background-light: #f9fafb;
    --border-color: #e5e7eb;
    --text-primary: #1f2937;
    --text-secondary: #6b7280;
}

/* Hide Gradio footer */
footer {
    display: none !important;
}

/* Main container */
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* Tabs */
.tab-nav button {
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
}

/* Chat */
.chatbot {
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
}

/* Chat messages - better readability */
.chatbot .message {
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.5rem !important;
}

.chatbot .bot {
    background: #f9fafb !important;
}

.chatbot .user {
    background: #475569 !important;
    color: white !important;
}

/* Buttons - calm style */
button.primary {
    background: #475569 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}

button.primary:hover {
    background: #334155 !important;
}

button.secondary {
    background: #f9fafb !important;
    border: 1px solid #d1d5db !important;
    color: #374151 !important;
    border-radius: 6px !important;
    font-weight: 400 !important;
}

button.secondary:hover {
    background: #f3f4f6 !important;
    border-color: #9ca3af !important;
}

/* Inputs */
textarea, input[type="text"] {
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #9ca3af !important;
    box-shadow: 0 0 0 2px rgba(156, 163, 175, 0.15) !important;
}

/* Quick buttons row - equal spacing */
.row {
    gap: 0.5rem !important;
}
"""


# Professional theme (Gradio 6.x - passed to launch())
# Calm, neutral colors
DEMO_THEME = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="#475569",
    button_primary_background_fill_hover="#334155",
    block_radius="8px",
    input_radius="6px",
)


# =============================================================================
# Build Gradio Interface (Gradio 6.x)
# =============================================================================
def create_demo() -> gr.Blocks:
    """Create Gradio demo interface."""
    
    with gr.Blocks(title="Product AI Assistant | Demo") as demo:
        
        # Clean Header (white background)
        gr.HTML("""
            <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 1rem; border-bottom: 1px solid #e5e7eb;">
                <h1 style="margin: 0; font-size: 1.5rem; font-weight: 600; color: #1f2937;">
                    Product AI Assistant
                </h1>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.9rem; color: #6b7280;">
                    Интеллектуальный консультант для карточки товара
                </p>
            </div>
        """)

        with gr.Tabs():
            # Tab 0: Product Chat (MAIN FEATURE - First!)
            with gr.TabItem("Консультант по товару"):
                
                with gr.Row():
                    # Left sidebar - clean settings
                    with gr.Column(scale=1, min_width=240):
                        chat_product_id = gr.Textbox(
                            label="Product ID",
                            placeholder="UUID товара",
                            value="62eb2515-1608-4812-9caa-12ad48c975c5",
                        )
                        
                        chat_store_id = gr.Textbox(
                            label="Store ID (опционально)",
                            placeholder="ID магазина",
                        )
                        
                        chat_conversation_id = gr.State("")
                        chat_suggested_questions = gr.State([])
                        
                        start_chat_btn = gr.Button("Начать консультацию", variant="primary", size="lg")
                        clear_btn = gr.Button("Очистить", variant="secondary")
                    
                    # Main chat area
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Диалог",
                            height=420,
                        )
                        
                        # Quick question buttons - 4 buttons to fit in one row
                        with gr.Row():
                            quick_q1 = gr.Button("Состав", size="sm", variant="secondary")
                            quick_q2 = gr.Button("Применение", size="sm", variant="secondary")
                            quick_q3 = gr.Button("Противопоказания", size="sm", variant="secondary")
                            quick_q4 = gr.Button("Побочные", size="sm", variant="secondary")
                        with gr.Row():
                            quick_q5 = gr.Button("Рецепт", size="sm", variant="secondary")
                            quick_q6 = gr.Button("Беременность", size="sm", variant="secondary")
                            quick_q7 = gr.Button("Хранение", size="sm", variant="secondary")
                            quick_q8 = gr.Button("Срок годности", size="sm", variant="secondary")
                        
                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="",
                                placeholder="Введите вопрос о товаре...",
                                lines=1,
                                scale=5,
                                container=False,
                            )
                            chat_btn = gr.Button("→", variant="primary", scale=1, min_width=60)

                def handle_start_chat(product_id, store_id):
                    """Initialize chat and show greeting with AI summary."""
                    if not product_id.strip():
                        return [{"role": "assistant", "content": "❌ Укажите Product ID"}], "", []
                    
                    history, conv_id = init_product_chat(product_id, store_id)
                    if not history:
                        return [{"role": "assistant", "content": "❌ Ошибка инициализации чата"}], "", []
                    
                    return history, conv_id, []

                def handle_chat(message, history, product_id, store_id, conv_id):
                    new_history, new_conv_id = chat_with_product(
                        message, history, product_id, store_id, conv_id
                    )
                    return new_history, "", new_conv_id

                def handle_quick_question(question, history, product_id, store_id, conv_id):
                    """Handle quick question button click."""
                    return handle_chat(question, history, product_id, store_id, conv_id)

                # Start chat button
                start_chat_btn.click(
                    handle_start_chat,
                    inputs=[chat_product_id, chat_store_id],
                    outputs=[chatbot, chat_conversation_id, chat_suggested_questions],
                )

                # Send message button
                chat_btn.click(
                    handle_chat,
                    inputs=[chat_input, chatbot, chat_product_id, chat_store_id, chat_conversation_id],
                    outputs=[chatbot, chat_input, chat_conversation_id],
                )
                
                # Quick question buttons
                for btn, question in [
                    (quick_q1, "Какой состав?"),
                    (quick_q2, "Как принимать?"),
                    (quick_q3, "Есть ли противопоказания?"),
                    (quick_q4, "Какие побочные эффекты?"),
                    (quick_q5, "Нужен ли рецепт?"),
                    (quick_q6, "Можно беременным?"),
                    (quick_q7, "Как хранить?"),
                    (quick_q8, "Какой срок годности?"),
                ]:
                    btn.click(
                        lambda h, p, s, c, q=question: handle_quick_question(q, h, p, s, c),
                        inputs=[chatbot, chat_product_id, chat_store_id, chat_conversation_id],
                        outputs=[chatbot, chat_input, chat_conversation_id],
                    )
                chat_input.submit(
                    handle_chat,
                    inputs=[chat_input, chatbot, chat_product_id, chat_store_id, chat_conversation_id],
                    outputs=[chatbot, chat_input, chat_conversation_id],
                )
                
                def do_clear():
                    return [], "", ""
                
                clear_btn.click(do_clear, outputs=[chatbot, chat_input, chat_conversation_id])

            # Tab 1: Main Assistant Chat (General pharmacy assistant)
            with gr.TabItem("Ассистент"):
                gr.Markdown("Универсальный помощник: поиск товаров, заказы, корзина.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        main_user_id = gr.Textbox(
                            label="User ID",
                            placeholder="demo-user",
                            value="demo-user",
                        )
                        main_conversation_id = gr.State("")
                        main_clear_btn = gr.Button("Очистить", variant="secondary")
                        
                        with gr.Accordion("Debug", open=False):
                            main_debug_output = gr.Markdown(
                                value="*Отправьте сообщение...*",
                            )
                    
                    with gr.Column(scale=3):
                        main_chatbot = gr.Chatbot(label="Диалог", height=380)
                        
                        gr.Markdown("*Примеры: \"болит голова\", \"найди нурофен\", \"покажи корзину\"*")
                        
                        with gr.Row():
                            main_input = gr.Textbox(
                                label="Ваш запрос",
                                placeholder="Введите запрос...",
                                lines=1,
                                scale=4,
                            )
                            main_chat_btn = gr.Button("Отправить", variant="primary", scale=1)

                def handle_main_chat(message, history, user_id, conv_id):
                    new_history, new_conv_id, debug_info = chat_with_assistant(
                        message, history, user_id, conv_id
                    )
                    return new_history, "", new_conv_id, debug_info or "*Нет debug информации*"

                main_chat_btn.click(
                    handle_main_chat,
                    inputs=[main_input, main_chatbot, main_user_id, main_conversation_id],
                    outputs=[main_chatbot, main_input, main_conversation_id, main_debug_output],
                )
                main_input.submit(
                    handle_main_chat,
                    inputs=[main_input, main_chatbot, main_user_id, main_conversation_id],
                    outputs=[main_chatbot, main_input, main_conversation_id, main_debug_output],
                )
                
                def do_main_clear():
                    return [], "", "", "*Чат очищен*"
                
                main_clear_btn.click(
                    do_main_clear, 
                    outputs=[main_chatbot, main_input, main_conversation_id, main_debug_output]
                )

            # Tab 2: FAQ
            with gr.TabItem("📋 FAQ генератор"):
                gr.Markdown("Автоматическая генерация FAQ на основе данных товара.")
                
                with gr.Row():
                    faq_product_id = gr.Textbox(
                        label="Product ID",
                        placeholder="12345",
                        value="12345",
                    )
                    faq_refresh = gr.Checkbox(label="Принудительно обновить", value=False)
                    faq_btn = gr.Button("Сгенерировать FAQ", variant="primary")
                
                faq_output = gr.Markdown(label="Результат")
                
                faq_btn.click(generate_faq, inputs=[faq_product_id, faq_refresh], outputs=[faq_output])

            # Tab 3: Drug Interactions
            with gr.TabItem("💊 Взаимодействия"):
                gr.Markdown("Проверка взаимодействий между лекарственными препаратами.")
                
                drugs_input = gr.Textbox(
                    label="Препараты (через запятую)",
                    placeholder="ибупрофен, аспирин, варфарин",
                    lines=2,
                )
                drugs_btn = gr.Button("Проверить", variant="primary")
                drugs_output = gr.Markdown(label="Результат")
                
                drugs_btn.click(check_drug_interactions, inputs=[drugs_input], outputs=[drugs_output])

            # Tab 4: Analogs
            with gr.TabItem("🔄 Аналоги"):
                gr.Markdown("Поиск аналогов по МНН (действующему веществу).")
                
                with gr.Row():
                    analog_product_id = gr.Textbox(label="Product ID", placeholder="12345")
                    analog_inn = gr.Textbox(
                        label="МНН (действующее вещество)",
                        placeholder="ибупрофен",
                        value="ибупрофен",
                    )
                
                with gr.Row():
                    analog_price = gr.Number(label="Текущая цена (₽)", value=350)
                    analog_limit = gr.Slider(label="Макс. результатов", minimum=1, maximum=20, value=5, step=1)
                    analog_cheaper = gr.Checkbox(label="Только дешевле", value=True)
                
                analog_btn = gr.Button("Найти аналоги", variant="primary")
                analog_output = gr.Markdown(label="Результат")
                
                analog_btn.click(
                    find_analogs,
                    inputs=[analog_product_id, analog_inn, analog_price, analog_limit, analog_cheaper],
                    outputs=[analog_output],
                )

            # Tab 5: Course Calculator
            with gr.TabItem("📊 Калькулятор курса"):
                gr.Markdown("Расчёт количества упаковок на курс лечения.")
                
                with gr.Row():
                    course_product_id = gr.Textbox(label="Product ID", placeholder="12345")
                    course_units = gr.Number(label="Единиц в упаковке", value=30, precision=0)
                    course_dose = gr.Number(label="Доза за приём (ед.)", value=1, precision=0)
                
                with gr.Row():
                    course_frequency = gr.Dropdown(
                        label="Частота приёма",
                        choices=[
                            ("1 раз в день", "once_daily"),
                            ("2 раза в день", "twice_daily"),
                            ("3 раза в день", "three_times_daily"),
                            ("4 раза в день", "four_times_daily"),
                            ("Через день", "every_other_day"),
                            ("1 раз в неделю", "once_weekly"),
                        ],
                        value="once_daily",
                    )
                    course_days = gr.Number(label="Дней курса", value=30, precision=0)
                    course_reserve = gr.Slider(label="Запас (%)", minimum=0, maximum=50, value=10, step=5)
                
                course_price = gr.Number(label="Цена упаковки (₽, опционально)", value=0)
                course_btn = gr.Button("Рассчитать", variant="primary")
                course_output = gr.Markdown(label="Результат")
                
                course_btn.click(
                    calculate_course,
                    inputs=[course_product_id, course_units, course_dose, course_frequency, 
                            course_days, course_reserve, course_price],
                    outputs=[course_output],
                )

            # Tab 6: Personalization
            with gr.TabItem("👤 Персонализация"):
                gr.Markdown("Контекст персонализации на основе истории покупок.")
                
                with gr.Row():
                    pers_user_id = gr.Textbox(
                        label="User ID",
                        placeholder="user-123",
                        value="user-123",
                    )
                    pers_product_id = gr.Textbox(
                        label="Product ID (опционально)",
                        placeholder="12345",
                    )
                
                pers_btn = gr.Button("Получить контекст", variant="primary")
                pers_output = gr.Markdown(label="Результат")
                
                pers_btn.click(
                    get_personalization,
                    inputs=[pers_user_id, pers_product_id],
                    outputs=[pers_output],
                )

            # Tab 7: Proactive Hints
            with gr.TabItem("💡 Проактивные подсказки"):
                gr.Markdown("Подсказки для разных триггеров поведения пользователя.")
                
                with gr.Row():
                    hints_product_id = gr.Textbox(
                        label="Product ID",
                        placeholder="12345",
                        value="12345",
                    )
                    hints_trigger = gr.Dropdown(
                        label="Триггер",
                        choices=[
                            ("Время на странице", "time_on_page"),
                            ("Намерение уйти", "exit_intent"),
                            ("Глубина скролла", "scroll_depth"),
                            ("Сомнение у корзины", "cart_hesitation"),
                            ("Повторный визит", "return_visit"),
                        ],
                        value="time_on_page",
                    )
                    hints_limit = gr.Slider(label="Макс. подсказок", minimum=1, maximum=10, value=3, step=1)
                
                hints_btn = gr.Button("Получить подсказки", variant="primary")
                hints_output = gr.Markdown(label="Результат")
                
                hints_btn.click(
                    get_proactive_hints,
                    inputs=[hints_product_id, hints_trigger, hints_limit],
                    outputs=[hints_output],
                )

            # Tab 8: Logs
            with gr.TabItem("📊 Логи"):
                gr.Markdown(
                    """
                    ## Журнал запросов и обработки
                    
                    Здесь отображаются логи всех запросов из Gradio с детальной информацией о работе пайплайна:
                    - **REQUEST/RESPONSE** — входящие запросы и ответы API
                    - **PIPELINE** — путь обработки (router_only, router+llm, llm_only и т.д.)
                    - **INTENT** — распознанные интенты
                    - **ROUTER** — детали матчинга роутера (триггеры, confidence)
                    - **LLM** — детали работы LLM (confidence, reasoning)
                    - **SLOTS** — извлечённые сущности
                    """
                )
                
                with gr.Row():
                    logs_filter = gr.Dropdown(
                        label="Фильтр по типу",
                        choices=[
                            ("Все", "ALL"),
                            ("Запросы/ответы", "REQUEST"),
                            ("Пайплайн", "PIPELINE"),
                            ("Интенты", "INTENT"),
                            ("Роутер", "ROUTER"),
                            ("LLM", "LLM"),
                            ("Слоты", "SLOTS"),
                            ("Контекст товара", "CONTEXT"),
                            ("Политики", "POLICY"),
                            ("Цитаты/поля", "CITATIONS"),
                            ("Ошибки", "ERROR"),
                        ],
                        value="ALL",
                    )
                    logs_limit = gr.Slider(
                        label="Количество записей",
                        minimum=10,
                        maximum=200,
                        value=50,
                        step=10,
                    )
                
                with gr.Row():
                    logs_refresh_btn = gr.Button("🔄 Обновить", variant="primary")
                    logs_clear_btn = gr.Button("🗑️ Очистить")
                    logs_json_btn = gr.Button("📋 JSON (debug)")
                
                logs_output = gr.Markdown(
                    value="📭 Логов пока нет. Отправьте запрос в любую вкладку.",
                    label="Логи",
                )
                logs_json_output = gr.Code(
                    label="JSON Debug",
                    language="json",
                    visible=False,
                )
                
                def refresh_logs(filter_level: str, limit: int) -> str:
                    return log_collector.format_as_text(limit=int(limit), filter_level=filter_level)
                
                def clear_logs() -> str:
                    log_collector.clear()
                    return "📭 Логи очищены."
                
                def show_json_logs(limit: int) -> tuple[gr.update, str]:
                    return gr.update(visible=True), log_collector.format_as_json(limit=int(limit))
                
                logs_refresh_btn.click(
                    refresh_logs,
                    inputs=[logs_filter, logs_limit],
                    outputs=[logs_output],
                )
                logs_clear_btn.click(clear_logs, outputs=[logs_output])
                logs_json_btn.click(
                    show_json_logs,
                    inputs=[logs_limit],
                    outputs=[logs_json_output, logs_json_output],
                )
                
                # Auto-refresh on filter/limit change
                logs_filter.change(
                    refresh_logs,
                    inputs=[logs_filter, logs_limit],
                )

            # Tab 9: LLM Debug (detailed prompts/responses)
            with gr.TabItem("🔬 LLM Debug"):
                gr.Markdown(
                    """
                    ## Детальная информация о вызовах LLM
                    
                    Здесь можно посмотреть **полные промпты** и **ответы** от LLM:
                    - Что именно отправляется в модель (system prompt + context + вопрос)
                    - Сырой ответ от модели
                    - Распарсенный результат
                    - Токены и latency
                    """
                )
                
                with gr.Row():
                    llm_debug_refresh_btn = gr.Button("🔄 Загрузить LLM вызовы", variant="primary")
                    llm_debug_clear_btn = gr.Button("🗑️ Очистить историю")
                
                llm_debug_selector = gr.Dropdown(
                    label="Выберите вызов для просмотра",
                    choices=[],
                    interactive=True,
                )
                
                with gr.Tabs():
                    with gr.TabItem("📊 Сводка"):
                        llm_debug_summary = gr.Markdown("*Нажмите 'Загрузить LLM вызовы'*")
                    
                    with gr.TabItem("📝 Полный промпт"):
                        llm_debug_prompt = gr.Textbox(
                            label="Full Prompt",
                            lines=20,
                            max_lines=50,
                            value="",
                        )
                    
                    with gr.TabItem("📥 Сырой ответ"):
                        llm_debug_raw_response = gr.Code(
                            label="Raw Response",
                            language="json",
                            value="",
                        )
                    
                    with gr.TabItem("✅ Распарсенный результат"):
                        llm_debug_parsed = gr.Code(
                            label="Parsed Response",
                            language="json",
                            value="",
                        )
                
                def fetch_llm_calls():
                    """Fetch LLM calls from API."""
                    result = api_call("GET", "/api/product-ai/debug/llm-calls?limit=20")
                    if "error" in result:
                        return gr.update(choices=[]), f"❌ {result['error']}"
                    
                    records = result.get("records", [])
                    if not records:
                        return gr.update(choices=[]), "📭 Нет записей о вызовах LLM. Отправьте запрос в 'Чат с товаром'."
                    
                    choices = []
                    for i, r in enumerate(reversed(records)):  # Newest first
                        ts = r.get("timestamp", "")
                        call_type = r.get("call_type", "")
                        msg_preview = r.get("user_message", "")[:30]
                        tokens = r.get("token_usage", {})
                        total = tokens.get("total_tokens", 0)
                        choices.append((f"[{ts}] {call_type}: \"{msg_preview}...\" ({total} tok)", i))
                    
                    summary = f"**Найдено вызовов:** {len(records)}\n\n"
                    for i, r in enumerate(reversed(records)):
                        ts = r.get("timestamp", "")
                        call_type = r.get("call_type", "")
                        model = r.get("model", "")
                        tokens = r.get("token_usage", {})
                        latency = r.get("latency_ms", 0)
                        cached = "✓" if r.get("cached") else "✗"
                        error = r.get("error")
                        
                        summary += f"**{i+1}. [{ts}] {call_type}**\n"
                        summary += f"- Model: `{model}` | Cached: {cached}\n"
                        summary += f"- Tokens: {tokens.get('prompt_tokens', 0)}→{tokens.get('completion_tokens', 0)} | Latency: {latency:.0f}ms\n"
                        if error:
                            summary += f"- ❌ Error: {error}\n"
                        summary += "\n"
                    
                    return gr.update(choices=choices, value=0 if choices else None), summary
                
                def show_llm_call_details(selected_idx):
                    """Show details for selected LLM call."""
                    if selected_idx is None:
                        return "", "", ""
                    
                    result = api_call("GET", "/api/product-ai/debug/llm-calls?limit=20")
                    records = result.get("records", [])
                    if not records:
                        return "", "", ""
                    
                    # Reverse to match dropdown order (newest first)
                    records = list(reversed(records))
                    if selected_idx >= len(records):
                        return "", "", ""
                    
                    r = records[selected_idx]
                    
                    full_prompt = r.get("full_prompt", r.get("system_prompt_full", ""))
                    raw_response = r.get("raw_response", "")
                    parsed = r.get("parsed_response", {})
                    
                    return (
                        full_prompt,
                        raw_response,
                        json.dumps(parsed, ensure_ascii=False, indent=2),
                    )
                
                def clear_llm_history():
                    api_call("DELETE", "/api/product-ai/debug/llm-calls")
                    return gr.update(choices=[]), "📭 История очищена."
                
                llm_debug_refresh_btn.click(
                    fetch_llm_calls,
                    outputs=[llm_debug_selector, llm_debug_summary],
                )
                
                llm_debug_clear_btn.click(
                    clear_llm_history,
                    outputs=[llm_debug_selector, llm_debug_summary],
                )
                
                llm_debug_selector.change(
                    show_llm_call_details,
                    inputs=[llm_debug_selector],
                    outputs=[llm_debug_prompt, llm_debug_raw_response, llm_debug_parsed],
                )

        # Footer - centered
        gr.HTML("""
            <div style="text-align: center; padding: 1rem 0; margin-top: 1rem; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 0.85rem;">
                <strong>Product AI Assistant</strong> — Демо &nbsp;•&nbsp; 
                <a href="http://127.0.0.1:8000/docs" style="color: #475569;">API Docs</a>
            </div>
        """)

    return demo


if __name__ == "__main__":
    import socket
    
    def find_free_port(start=7860, end=7870):
        for port in range(start, end):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", port))
                s.close()
                return port
            except OSError:
                continue
        return start
    
    print("Starting Gradio demo...")
    print(f"API endpoint: {API_BASE}")
    port = find_free_port()
    print(f"Using port: {port}")
    demo = create_demo()

    # Publishing options:
    # - GRADIO_SHARE=true -> creates a public URL (tunnel) while this process runs
    # - GRADIO_AUTH=user:pass (or multiple "u:p,u2:p2") -> basic auth for the UI
    share = _env_bool("GRADIO_SHARE", False)
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    auth = _parse_gradio_auth(os.getenv("GRADIO_AUTH"))

    demo.launch(
        server_name=server_name,
        server_port=port,
        share=share,
        auth=auth,
        show_error=True,
        theme=DEMO_THEME,
        css=CUSTOM_CSS,
    )
