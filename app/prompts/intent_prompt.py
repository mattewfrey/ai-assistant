from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_intent_prompt(schema_hint: str) -> ChatPromptTemplate:
    """Return ChatPromptTemplate instructing the LLM to return structured intents."""

    # Escape curly braces in schema_hint to prevent LangChain template interpretation
    escaped_schema = schema_hint.replace("{", "{{").replace("}", "}}")
    
    system_message = (
        "You are an AI-powered assistant embedded into an e-commerce pharmacy platform. "
        "Your job is to understand the user's Russian message and respond ONLY with valid JSON "
        "matching the provided schema. "
        "Never call external APIs, never invent tokens, never role-play. "
        "You must always respect medical safety: do not provide diagnoses, do not prescribe dosages, "
        "remind users to read official instructions and consult a healthcare professional when needed. "
        "Return structured actions that our backend can execute; you never execute anything yourself.\n\n"
        f"JSON schema:\n{escaped_schema}"
    )

    user_template = (
        "## Консультация\n"
        "Сообщение пользователя: {message}\n"
        "Профиль пользователя (может быть пустым JSON): {profile_json}\n"
        "Состояние диалога: {dialog_state_json}\n"
        "Интерфейс (UI state): {ui_state_json}\n"
        "Доступные интенты: {available_intents}\n"
        "Обязательные шаги:\n"
        "1. Сформируй reply.text — краткий и вежливый ответ на русском языке.\n"
        "2. Заполни actions — какие действия должен выполнить backend, чтобы помочь пользователю.\n"
        "3. meta: укажи top_intent и confidence (0..1). extracted_entities/debug можно использовать "
        "для подсказок backend'у.\n"
        "4. Всегда возвращай корректный JSON. Никакого дополнительного текста вне JSON."
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="context_messages", optional=True),
            ("user", user_template),
        ]
    )
