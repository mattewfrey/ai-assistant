from __future__ import annotations

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


def test_chat_slot_followup_flow(client: TestClient):
    first = client.post("/api/ai/chat/message", json={"message": "болит голова до 500 рублей"})
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body["data"]["products"] == []
    assert "возраст" in first_body["reply"]["text"].lower() or "сколько" in first_body["reply"]["text"].lower()
    conv_id = first_body["conversation_id"]

    second = client.post(
        "/api/ai/chat/message",
        json={"message": "мне 30", "conversation_id": conv_id},
    )
    assert second.status_code == 200, second.text
    body2 = second.json()
    assert body2["data"]["products"]
    debug = body2["meta"]["debug"]
    assert debug["slot_filling_used"] is True
    assert debug["llm_used"] is False

