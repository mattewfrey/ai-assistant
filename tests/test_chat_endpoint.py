from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_chat_endpoint_show_cart_flow(client: TestClient) -> None:
    """Navigation request should be handled by router without LLM."""
    response = client.post("/api/ai/chat/message", json={"message": "Покажи корзину"})

    assert response.status_code == 200, response.text
    payload = response.json()

    assert (payload.get("reply", {}) or {}).get("text")
    intents = {action.get("intent") for action in payload.get("actions") or []}
    assert "SHOW_CART" in intents
    assert payload.get("data", {}).get("cart") is not None
    debug = payload.get("meta", {}).get("debug") or {}
    assert debug.get("llm_used") is False


def test_chat_endpoint_symptom_with_price_filter(client: TestClient) -> None:
    """Symptom query should return filtered products from mock platform."""
    response = client.post(
        "/api/ai/chat/message",
        json={"message": "Болит голова, нужны таблетки до 500р"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()

    # Может вернуть продукты или спросить уточнение
    products = payload.get("data", {}).get("products") or []
    if products:
        prices = [p.get("price") for p in products if p.get("price") is not None]
        assert all(price <= 500 for price in prices), "Все товары должны быть до 500р"
    
    # Проверяем что есть ответ
    reply_text = (payload.get("reply", {}) or {}).get("text", "").lower()
    assert reply_text, "Ожидали текстовый ответ"
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app


@pytest.fixture
def test_settings():
    return Settings(
        openai_api_key="",
        use_langchain=False,
        enable_beautify_reply=False,
    )


@pytest.fixture
def client(test_settings, monkeypatch):
    def _get_settings_override():
        return test_settings

    app = create_app()
    app.dependency_overrides[get_settings] = _get_settings_override
    return TestClient(app)


def test_chat_cart_router_path(client: TestClient):
    payload = {"message": "покажи корзину"}
    resp = client.post("/api/ai/chat/message", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["meta"]["debug"]["router_matched"] is True
    assert body["meta"]["debug"]["llm_used"] is False
    assert body["meta"]["debug"]["source"] == "local_router"


def test_chat_slot_followup_flow(client: TestClient):
    """Тест флоу с уточнением слотов: age_group обязателен для FIND_BY_SYMPTOM."""
    first = client.post("/api/ai/chat/message", json={"message": "болит голова до 500 рублей"})
    assert first.status_code == 200, first.text
    first_body = first.json()
    
    # Теперь age_group обязателен, поэтому должен спрашивать
    reply_text = first_body["reply"]["text"].lower()
    # Проверяем вопрос про возрастную группу
    assert any(word in reply_text for word in ["кого", "взрослый", "ребёнок", "ребенок", "подросток", "пожилой", "возраст", "лет"]), \
        f"Должен спросить про возрастную группу, получено: {reply_text}"
    
    # Проверяем debug флаги
    debug = first_body["meta"]["debug"]
    assert debug.get("slot_filling_used") is True
    assert debug.get("slot_prompt_pending") is True
    
    conv_id = first_body["conversation_id"]
    second = client.post(
        "/api/ai/chat/message",
        json={"message": "для взрослого", "conversation_id": conv_id},
    )
    assert second.status_code == 200, second.text
    body2 = second.json()
    # После указания возрастной группы должны вернуться продукты
    assert body2["data"]["products"], "После указания возрастной группы должны вернуться продукты"
    debug2 = body2["meta"]["debug"]
    assert debug2["slot_filling_used"] is True
    assert debug2.get("slot_prompt_pending") is False

