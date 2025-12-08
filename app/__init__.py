"""
Assistant Gateway package.

This module exposes helpers for creating the FastAPI application so that both
runtime code and tests can import the same app instance.
"""

from .main import create_app

__all__ = ["create_app"]

