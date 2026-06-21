"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., examples=["user"])
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 500

    class Config:
        extra = "allow"


class CompletionMetadata(BaseModel):
    request_id: int | None = None
    selected_model_key: str
    selected_model_id: str
    provider: str
    complexity_tier: str
    routing_reason: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    baseline_model_key: str
    baseline_estimated_cost: float
    estimated_savings: float
    estimated_savings_percent: float
    verification_queued_or_sampled: bool
    verification_status: str


class CompletionResponse(BaseModel):
    response: str
    metadata: CompletionMetadata


class EvaluateRequest(BaseModel):
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    response: str
    request_id: int | None = None
    use_llm_judge: bool = False


class EvaluationResponse(BaseModel):
    quality_score: float
    passed: bool
    failure_reason: str | None
    verifier_model_key: str | None = None
    verifier_cost: float = 0.0
    checks: dict[str, Any]


class RoutingConfigUpdateResponse(BaseModel):
    message: str
    routing_config: dict[str, Any]

