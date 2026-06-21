"""Provider factory."""

from __future__ import annotations

from app.config_loader import is_mock_mode
from app.providers.base import BaseLLMProvider
from app.providers.mock_provider import MockProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.openrouter_provider import OpenRouterProvider


def get_provider(provider_name: str, force_mock: bool | None = None) -> BaseLLMProvider:
    mock_mode = is_mock_mode() if force_mock is None else force_mock
    if mock_mode:
        return MockProvider()

    if provider_name == "openai":
        return OpenAIProvider()
    if provider_name == "openrouter":
        return OpenRouterProvider()

    raise ValueError(f"Unsupported provider '{provider_name}'.")

