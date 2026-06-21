"""Cost calculation helpers.

All prices are read from model configuration. The application code never
hard-codes model prices.
"""

from __future__ import annotations


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model_config: dict,
) -> float:
    """Calculate cost from token counts and per-1M token pricing."""
    input_cost_per_1m = float(model_config.get("input_cost_per_1m", 0.0))
    output_cost_per_1m = float(model_config.get("output_cost_per_1m", 0.0))

    input_cost = input_tokens / 1_000_000 * input_cost_per_1m
    output_cost = output_tokens / 1_000_000 * output_cost_per_1m
    return round(input_cost + output_cost, 10)


def estimate_baseline_cost(
    input_tokens: int,
    output_tokens: int,
    baseline_model_config: dict,
) -> float:
    """Estimate cost if the same request had used the baseline model."""
    return calculate_cost(input_tokens, output_tokens, baseline_model_config)


def calculate_savings(actual_cost: float, baseline_cost: float) -> tuple[float, float]:
    """Return absolute and percentage savings versus baseline."""
    savings = round(baseline_cost - actual_cost, 10)
    if baseline_cost <= 0:
        return savings, 0.0

    savings_percent = round((savings / baseline_cost) * 100, 2)
    return savings, savings_percent

