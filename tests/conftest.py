"""Shared pytest fixtures for all tests."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.models import ChatRequest, UIState, UserPreferences, UserProfile
from app.services.mock_platform import MockPlatform
from app.services.platform_client import PlatformApiClient


@pytest.fixture
def settings() -> Settings:
    """Default settings for tests."""
    return Settings(openai_api_key="", use_langchain=False)


@pytest.fixture
def platform_client(settings: Settings) -> PlatformApiClient:
    """Platform client instance."""
    return PlatformApiClient(settings)


@pytest.fixture
def mock_platform() -> MockPlatform:
    """Mock platform instance."""
    return MockPlatform()


@pytest.fixture
def basic_request() -> ChatRequest:
    """Basic chat request without user."""
    return ChatRequest(conversation_id="test-conv", message="тест")


@pytest.fixture
def user_request() -> ChatRequest:
    """Chat request with user_id."""
    return ChatRequest(
        conversation_id="test-conv",
        message="тест",
        user_id="user-test",
    )


@pytest.fixture
def request_with_pharmacy() -> ChatRequest:
    """Chat request with selected pharmacy."""
    return ChatRequest(
        conversation_id="test-conv",
        message="тест",
        user_id="user-test",
        ui_state=UIState(selected_pharmacy_id="ph1"),
    )


@pytest.fixture
def basic_profile() -> UserProfile:
    """Basic user profile."""
    return UserProfile(
        user_id="user-test",
        preferences=UserPreferences(),
    )


@pytest.fixture
def profile_with_children() -> UserProfile:
    """User profile with has_children=True."""
    return UserProfile(
        user_id="user-test",
        preferences=UserPreferences(has_children=True),
    )


@pytest.fixture
def profile_with_preferences() -> UserProfile:
    """User profile with various preferences."""
    return UserProfile(
        user_id="user-test",
        preferences=UserPreferences(
            sugar_free=True,
            lactose_free=True,
            for_children=False,
            age=35,
            region="MOW",
            default_max_price=1000,
            preferred_forms=["таблетки", "капсулы"],
        ),
    )


@pytest.fixture
def sample_product_id() -> str:
    """Sample product ID from catalog."""
    return "prod-theraflu"


@pytest.fixture
def sample_pharmacy_id() -> str:
    """Sample pharmacy ID."""
    return "ph1"

