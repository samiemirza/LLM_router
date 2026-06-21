"""Provider interface and shared response object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    provider: str
    model_id: str
    input_tokens: int
    output_tokens: int
    total_cost: float
    latency_ms: float
    raw_response: Any


class ProviderError(Exception):
    """Raised when an LLM provider call fails."""


class BaseLLMProvider:
    provider_name: str

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model_config: dict,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        raise NotImplementedError

