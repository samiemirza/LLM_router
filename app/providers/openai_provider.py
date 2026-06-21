"""OpenAI provider wrapper."""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from app.cost import calculate_cost
from app.providers.base import BaseLLMProvider, ProviderError, ProviderResponse


class OpenAIProvider(BaseLLMProvider):
    provider_name = "openai"

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is required when LLM_MOCK_MODE is false.")
        self.client = OpenAI(api_key=api_key)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model_config: dict,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        model_id = str(model_config["model_id"])
        start_time = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI provider error for model '{model_id}': {exc}") from exc

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        text = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_cost = calculate_cost(input_tokens, output_tokens, model_config)

        return ProviderResponse(
            text=text,
            provider="openai",
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost=total_cost,
            latency_ms=latency_ms,
            raw_response=_serialize_response(response),
        )


def _serialize_response(response: Any) -> Any:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return str(response)
