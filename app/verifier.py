"""Quality verification helpers."""

from __future__ import annotations

import json
import random
import re
from typing import Any

from app.config_loader import is_mock_mode
from app.features import get_prompt_text
from app.providers import get_provider
from app.providers.base import ProviderError


def should_sample_verification(routing_config: dict[str, Any]) -> bool:
    verification_config = routing_config.get("verification", {})
    if not verification_config.get("enabled", False):
        return False
    if verification_config.get("mode", "sampled") != "sampled":
        return False

    sample_rate = float(verification_config.get("sample_rate", 0.0))
    return random.random() < sample_rate


def get_verification_status(routing_config: dict[str, Any]) -> str:
    if not routing_config.get("verification", {}).get("enabled", False):
        return "manual_only"
    return "sampled_pending" if should_sample_verification(routing_config) else "not_sampled"


def evaluate_response(
    messages: list[dict[str, str]],
    response_text: str,
    routing_config: dict[str, Any],
    models_config: dict[str, Any],
    use_llm_judge: bool = False,
) -> dict[str, Any]:
    """Run deterministic checks and optionally an LLM judge."""
    deterministic_result = run_deterministic_checks(messages, response_text, routing_config)
    if not use_llm_judge:
        return deterministic_result

    if is_mock_mode():
        deterministic_result["checks"]["llm_judge"] = "skipped_in_mock_mode"
        return deterministic_result

    try:
        llm_result = run_llm_judge(messages, response_text, routing_config, models_config)
    except ProviderError as exc:
        deterministic_result["checks"]["llm_judge_error"] = str(exc)
        return deterministic_result

    return llm_result


def run_deterministic_checks(
    messages: list[dict[str, str]],
    response_text: str,
    routing_config: dict[str, Any],
) -> dict[str, Any]:
    prompt = get_prompt_text(messages)
    lowered_prompt = prompt.lower()
    checks: dict[str, Any] = {}
    failures: list[str] = []

    checks["non_empty_output"] = bool(response_text.strip())
    if not checks["non_empty_output"]:
        failures.append("Output is empty.")

    if "json" in lowered_prompt:
        try:
            json.loads(response_text)
            checks["valid_json"] = True
        except json.JSONDecodeError:
            checks["valid_json"] = False
            failures.append("Prompt requested JSON, but output was not valid JSON.")

    if "bullet" in lowered_prompt or "bullets" in lowered_prompt:
        bullet_like = bool(re.search(r"(^|\n)\s*(-|\*|\d+[.)])\s+", response_text))
        checks["bullet_structure"] = bullet_like
        if not bullet_like:
            failures.append("Prompt requested bullets, but output did not contain bullet structure.")

    if "return only" in lowered_prompt:
        checks["return_only_rough_check"] = len(response_text.splitlines()) <= 3
        if not checks["return_only_rough_check"]:
            failures.append("Prompt requested a concise return-only answer.")

    pass_threshold = float(routing_config.get("verification", {}).get("pass_threshold", 4.0))
    quality_score = 5.0 if not failures else max(1.0, 5.0 - len(failures))
    passed = quality_score >= pass_threshold

    return {
        "quality_score": quality_score,
        "passed": passed,
        "failure_reason": "; ".join(failures) if failures else None,
        "verifier_model_key": None,
        "verifier_cost": 0.0,
        "checks": checks,
    }


def run_llm_judge(
    messages: list[dict[str, str]],
    response_text: str,
    routing_config: dict[str, Any],
    models_config: dict[str, Any],
) -> dict[str, Any]:
    verifier_model_key = routing_config.get("verification", {}).get("verifier_model")
    if verifier_model_key not in models_config:
        raise ProviderError(f"Verifier model key '{verifier_model_key}' is not defined.")

    verifier_config = models_config[verifier_model_key]
    provider = get_provider(str(verifier_config["provider"]))
    judge_prompt = _build_judge_prompt(messages, response_text)
    provider_response = provider.chat_completion(
        messages=[{"role": "user", "content": judge_prompt}],
        model_config=verifier_config,
        temperature=0.0,
        max_tokens=250,
    )

    try:
        parsed = json.loads(provider_response.text)
        score = float(parsed.get("score", 0.0))
        reason = parsed.get("failure_reason")
    except (json.JSONDecodeError, TypeError, ValueError):
        score = 0.0
        reason = "LLM judge did not return valid JSON."

    pass_threshold = float(routing_config.get("verification", {}).get("pass_threshold", 4.0))
    return {
        "quality_score": score,
        "passed": score >= pass_threshold,
        "failure_reason": reason,
        "verifier_model_key": verifier_model_key,
        "verifier_cost": provider_response.total_cost,
        "checks": {"llm_judge": True},
    }


def _build_judge_prompt(messages: list[dict[str, str]], response_text: str) -> str:
    prompt = get_prompt_text(messages)
    return (
        "You are judging whether an assistant response satisfies the user's request.\n"
        "Return only JSON with keys: score, passed, failure_reason.\n"
        "Score from 1 to 5, where 5 means excellent.\n\n"
        f"User request:\n{prompt}\n\n"
        f"Assistant response:\n{response_text}"
    )

