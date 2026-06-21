import os

from app.config_loader import load_models_config, load_routing_config
from app.providers.mock_provider import MockProvider
from app.router import get_baseline_model, select_model_for_tier, validate_routing_config


def test_routing_config_points_to_defined_models() -> None:
    models_config = load_models_config()
    routing_config = load_routing_config()

    validate_routing_config(routing_config, models_config)
    baseline_key, baseline_config = get_baseline_model(routing_config, models_config)

    assert baseline_key in models_config
    assert "model_id" in baseline_config


def test_select_model_for_simple_tier() -> None:
    models_config = load_models_config()
    routing_config = load_routing_config()

    decision = select_model_for_tier("simple", routing_config, models_config, "test reason")

    assert decision.selected_model_key == routing_config["routing"]["simple"]["primary_model"]
    assert decision.selected_model_config["provider"] in {"openai", "openrouter"}


def test_mock_provider_returns_standard_response(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    provider = MockProvider()
    model_config = load_models_config()["deepseek_v4_flash"]

    response = provider.chat_completion(
        messages=[{"role": "user", "content": "Return JSON with status ok."}],
        model_config=model_config,
        temperature=0.0,
        max_tokens=100,
    )

    assert response.text.startswith("{")
    assert response.input_tokens > 0
    assert response.total_cost >= 0
    assert os.getenv("LLM_MOCK_MODE") == "true"

