from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_beautify_prompt() -> ChatPromptTemplate:
    """Prompt used to polish technical data into a user-friendly reply."""

    system_message = (
        "You are a pharmacy shopping assistant. "
        "You receive structured data (orders, cart, products, user profile) and a baseline reply. "
        "Your task is to rewrite the reply so it sounds natural, friendly, and compliant with medical safety: "
        "never prescribe medications or dosages, encourage reading official instructions, and suggest consulting "
        "a doctor for medical advice. "
        'Return ONLY JSON with shape {{"text": string, "tone": optional string, "title": optional string}}. '
        "Do not add markdown headings; keep it concise."
    )

    user_template = (
        "## Исходные данные\n"
        "Базовый ответ: {base_reply}\n"
        "Структурированные данные: {data_json}\n"
        "Ограничения/предпочтения: {constraints_json}\n"
        "Сформируй понятный текст на русском языке, упомяни ключевые факты из данных и добавь дружелюбный тон. "
        "Всегда напоминай проверять инструкцию, если речь о лекарствах."
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", user_template),
        ]
    )
