"""Model routing based on complexity tier and YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoutingDecision:
    complexity_tier: str
    selected_model_key: str
    selected_model_config: dict[str, Any]
    fallback_model_key: str | None
    routing_reason: str


def select_model_for_tier(
    complexity_tier: str,
    routing_config: dict[str, Any],
    models_config: dict[str, Any],
    routing_reason: str,
) -> RoutingDecision:
    """Choose a primary model for a classified complexity tier."""
    tier_config = routing_config.get("routing", {}).get(complexity_tier)
    if not tier_config:
        raise ValueError(f"No routing rule configured for tier '{complexity_tier}'.")

    model_key = tier_config.get("primary_model")
    fallback_model_key = tier_config.get("fallback_model")
    if model_key not in models_config:
        raise ValueError(f"Primary model key '{model_key}' is not defined in models.yaml.")

    if fallback_model_key and fallback_model_key not in models_config:
        raise ValueError(f"Fallback model key '{fallback_model_key}' is not defined in models.yaml.")

    return RoutingDecision(
        complexity_tier=complexity_tier,
        selected_model_key=model_key,
        selected_model_config=models_config[model_key],
        fallback_model_key=fallback_model_key,
        routing_reason=routing_reason,
    )


def get_baseline_model(
    routing_config: dict[str, Any],
    models_config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    baseline_key = routing_config.get("baseline", {}).get("model")
    if baseline_key not in models_config:
        raise ValueError(f"Baseline model key '{baseline_key}' is not defined in models.yaml.")
    return baseline_key, models_config[baseline_key]


def validate_routing_config(
    routing_config: dict[str, Any],
    models_config: dict[str, Any],
) -> None:
    for tier, tier_config in routing_config.get("routing", {}).items():
        primary_model = tier_config.get("primary_model")
        fallback_model = tier_config.get("fallback_model")
        if primary_model not in models_config:
            raise ValueError(f"Tier '{tier}' uses unknown primary model '{primary_model}'.")
        if fallback_model and fallback_model not in models_config:
            raise ValueError(f"Tier '{tier}' uses unknown fallback model '{fallback_model}'.")
    get_baseline_model(routing_config, models_config)

