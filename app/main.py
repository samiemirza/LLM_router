"""FastAPI entry point for LLM Cost Autopilot."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from app.classifier import classify_messages
from app.config_loader import (
    get_app_name,
    load_models_config,
    load_routing_config,
    update_runtime_routing_config,
)
from app.cost import calculate_savings, estimate_baseline_cost
from app.features import get_prompt_text
from app.logging_db import (
    get_recent_requests,
    get_stats,
    init_db,
    log_request,
    log_verification,
)
from app.providers import get_provider
from app.providers.base import ProviderError, ProviderResponse
from app.router import get_baseline_model, select_model_for_tier, validate_routing_config
from app.schemas import (
    ChatCompletionRequest,
    CompletionMetadata,
    CompletionResponse,
    EvaluateRequest,
    EvaluationResponse,
    RoutingConfigUpdateResponse,
)
from app.verifier import evaluate_response, get_verification_status


app = FastAPI(title=get_app_name())


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/completions", response_model=CompletionResponse)
def completions(request: ChatCompletionRequest) -> CompletionResponse:
    try:
        return process_chat_completion(request)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def process_chat_completion(request: ChatCompletionRequest) -> CompletionResponse:
    models_config = load_models_config()
    routing_config = load_routing_config()
    validate_routing_config(routing_config, models_config)

    messages = _messages_to_dicts(request.messages)
    prompt = get_prompt_text(messages)
    classification = classify_messages(messages)
    decision = select_model_for_tier(
        classification.complexity_tier,
        routing_config,
        models_config,
        classification.routing_reason,
    )

    provider_response, selected_model_key, selected_model_config, routing_reason = _call_with_fallback(
        messages=messages,
        initial_model_key=decision.selected_model_key,
        fallback_model_key=decision.fallback_model_key,
        models_config=models_config,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        routing_reason=decision.routing_reason,
    )

    baseline_model_key, baseline_model_config = get_baseline_model(routing_config, models_config)
    baseline_cost = estimate_baseline_cost(
        provider_response.input_tokens,
        provider_response.output_tokens,
        baseline_model_config,
    )
    estimated_savings, estimated_savings_percent = calculate_savings(
        provider_response.total_cost,
        baseline_cost,
    )
    verification_status = get_verification_status(routing_config)

    request_id = log_request(
        prompt=prompt,
        complexity_tier=classification.complexity_tier,
        selected_model_key=selected_model_key,
        selected_model_id=str(selected_model_config["model_id"]),
        provider=str(selected_model_config["provider"]),
        routing_reason=routing_reason,
        input_tokens=provider_response.input_tokens,
        output_tokens=provider_response.output_tokens,
        cost=provider_response.total_cost,
        latency_ms=provider_response.latency_ms,
        response_text=provider_response.text,
        baseline_model_key=baseline_model_key,
        baseline_estimated_cost=baseline_cost,
        estimated_savings=estimated_savings,
        estimated_savings_percent=estimated_savings_percent,
        verification_status=verification_status,
    )

    return CompletionResponse(
        response=provider_response.text,
        metadata=CompletionMetadata(
            request_id=request_id,
            selected_model_key=selected_model_key,
            selected_model_id=str(selected_model_config["model_id"]),
            provider=str(selected_model_config["provider"]),
            complexity_tier=classification.complexity_tier,
            routing_reason=routing_reason,
            input_tokens=provider_response.input_tokens,
            output_tokens=provider_response.output_tokens,
            cost=provider_response.total_cost,
            latency_ms=provider_response.latency_ms,
            baseline_model_key=baseline_model_key,
            baseline_estimated_cost=baseline_cost,
            estimated_savings=estimated_savings,
            estimated_savings_percent=estimated_savings_percent,
            verification_queued_or_sampled=verification_status == "sampled_pending",
            verification_status=verification_status,
        ),
    )


def _call_with_fallback(
    *,
    messages: list[dict[str, str]],
    initial_model_key: str,
    fallback_model_key: str | None,
    models_config: dict[str, Any],
    temperature: float,
    max_tokens: int,
    routing_reason: str,
) -> tuple[ProviderResponse, str, dict[str, Any], str]:
    selected_config = models_config[initial_model_key]
    try:
        provider_response = _call_provider(
            messages,
            selected_config,
            temperature,
            max_tokens,
        )
        return provider_response, initial_model_key, selected_config, routing_reason
    except ProviderError as primary_error:
        if not fallback_model_key or fallback_model_key == initial_model_key:
            raise

        fallback_config = models_config[fallback_model_key]
        try:
            provider_response = _call_provider(
                messages,
                fallback_config,
                temperature,
                max_tokens,
            )
        except ProviderError:
            raise primary_error

        reason = (
            f"{routing_reason} Primary model '{initial_model_key}' failed; "
            f"used fallback '{fallback_model_key}'."
        )
        return provider_response, fallback_model_key, fallback_config, reason


def _call_provider(
    messages: list[dict[str, str]],
    model_config: dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> ProviderResponse:
    provider = get_provider(str(model_config["provider"]))
    return provider.chat_completion(
        messages=messages,
        model_config=model_config,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"models": load_models_config()}


@app.get("/v1/stats")
def stats() -> dict[str, Any]:
    return get_stats()


@app.get("/v1/recent-requests")
def recent_requests(limit: int = 50) -> dict[str, Any]:
    return {"requests": get_recent_requests(limit=limit)}


@app.put("/v1/routing-config", response_model=RoutingConfigUpdateResponse)
def update_routing_config(config_update: dict[str, Any]) -> RoutingConfigUpdateResponse:
    updated_config = update_runtime_routing_config(config_update)
    models_config = load_models_config()
    validate_routing_config(updated_config, models_config)
    return RoutingConfigUpdateResponse(
        message="Runtime routing config updated for the current process only.",
        routing_config=updated_config,
    )


@app.post("/v1/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluateRequest) -> EvaluationResponse:
    models_config = load_models_config()
    routing_config = load_routing_config()
    messages = _evaluation_messages(request)
    result = evaluate_response(
        messages=messages,
        response_text=request.response,
        routing_config=routing_config,
        models_config=models_config,
        use_llm_judge=request.use_llm_judge,
    )

    log_verification(
        request_id=request.request_id,
        verifier_model_key=result.get("verifier_model_key"),
        quality_score=float(result["quality_score"]),
        passed=bool(result["passed"]),
        failure_reason=result.get("failure_reason"),
        verifier_cost=float(result.get("verifier_cost", 0.0)),
    )

    return EvaluationResponse(**result)


def _messages_to_dicts(messages: list[Any]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _evaluation_messages(request: EvaluateRequest) -> list[dict[str, str]]:
    if request.messages:
        return _messages_to_dicts(request.messages)
    if request.prompt:
        return [{"role": "user", "content": request.prompt}]
    raise HTTPException(status_code=400, detail="Either prompt or messages must be provided.")

