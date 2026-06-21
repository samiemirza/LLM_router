from app.cost import calculate_cost, calculate_savings, estimate_baseline_cost


def test_calculate_cost_uses_per_million_prices() -> None:
    model_config = {"input_cost_per_1m": 1.0, "output_cost_per_1m": 10.0}

    cost = calculate_cost(1_000, 2_000, model_config)

    assert cost == 0.021


def test_estimate_baseline_cost_matches_calculate_cost() -> None:
    model_config = {"input_cost_per_1m": 5.0, "output_cost_per_1m": 30.0}

    assert estimate_baseline_cost(100, 50, model_config) == calculate_cost(
        100,
        50,
        model_config,
    )


def test_calculate_savings_returns_absolute_and_percent() -> None:
    savings, percent = calculate_savings(actual_cost=0.25, baseline_cost=1.0)

    assert savings == 0.75
    assert percent == 75.0

