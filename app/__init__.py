"""
Assistant Gateway package.

This module exposes helpers for creating the FastAPI application so that both
runtime code and tests can import the same app instance.

Note: We use lazy imports to avoid creating the FastAPI app on import,
which can cause issues with some Python/FastAPI version combinations.
"""


def create_app():
    """Lazy import wrapper for create_app to avoid import-time app creation."""
    from .main import create_app as _create_app
    return _create_app()


def get_app():
    """Get or create the FastAPI application instance."""
    from .main import app
    return app


__all__ = ["create_app", "get_app"]

