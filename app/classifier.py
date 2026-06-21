"""Complexity classification.

The app uses a trained scikit-learn model if one exists. Otherwise it falls
back to transparent rules that are easy to inspect and adjust.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.features import extract_features, features_to_vector, get_prompt_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "complexity_classifier.pkl"
VALID_TIERS = {"simple", "moderate", "complex"}


@dataclass(frozen=True)
class ClassificationResult:
    complexity_tier: str
    routing_reason: str
    classifier_source: str


def classify_messages(
    messages: list[dict[str, Any]],
    model_path: Path = DEFAULT_CLASSIFIER_PATH,
) -> ClassificationResult:
    """Classify messages into simple, moderate, or complex."""
    features = extract_features(messages)
    prompt = get_prompt_text(messages)

    if model_path.exists():
        result = _classify_with_trained_model(features, model_path)
        if result is not None:
            return result

    return classify_with_rules(features, prompt)


def _classify_with_trained_model(
    features: dict[str, int],
    model_path: Path,
) -> ClassificationResult | None:
    try:
        with model_path.open("rb") as file:
            saved_model = pickle.load(file)

        model = saved_model["model"] if isinstance(saved_model, dict) else saved_model
        prediction = model.predict([features_to_vector(features)])[0]
        tier = str(prediction)
        if tier not in VALID_TIERS:
            return None

        return ClassificationResult(
            complexity_tier=tier,
            routing_reason=f"Trained classifier selected {tier} from prompt features.",
            classifier_source="sklearn",
        )
    except Exception:
        return None


def classify_with_rules(
    features: dict[str, int],
    prompt: str,
) -> ClassificationResult:
    """Rule-based fallback classifier."""
    lowered = prompt.lower()
    token_count = features["approx_token_count"]
    constraints = features["estimated_constraint_count"]

    complex_terms = [
        "architecture",
        "system design",
        "debug",
        "root cause",
        "multi-step",
        "step by step",
        "strategy",
        "nuanced",
        "tradeoff",
        "trade-off",
        "optimize",
        "design a solution",
    ]
    moderate_terms = [
        "summarize",
        "summary",
        "classify",
        "categorize",
        "compare",
        "analysis",
        "analyze",
        "business",
        "email",
        "proposal",
        "table",
        "structured",
        "recommend",
    ]

    if token_count > 900:
        return ClassificationResult(
            "complex",
            "Long-context task likely needs stronger reasoning.",
            "rules",
        )

    if features["has_code"] and (
        "debug" in lowered or "fix" in lowered or "error" in lowered or token_count > 250
    ):
        return ClassificationResult(
            "complex",
            "Code debugging or implementation task detected.",
            "rules",
        )

    if any(term in lowered for term in complex_terms) or (
        features["has_reasoning_keywords"] and constraints >= 4
    ):
        return ClassificationResult(
            "complex",
            "Multi-step reasoning, architecture, or judgment-heavy task detected.",
            "rules",
        )

    if token_count <= 140 and features["has_simple_task_keywords"] and constraints <= 4:
        return ClassificationResult(
            "simple",
            "Short rewriting, extraction, cleanup, or formatting task.",
            "rules",
        )

    if token_count <= 90 and (
        features["has_json_request"]
        or features["has_bullet_request"]
        or features["has_table_request"]
    ):
        return ClassificationResult(
            "simple",
            "Short structured formatting request.",
            "rules",
        )

    if (
        any(term in lowered for term in moderate_terms)
        or features["has_comparison_keywords"]
        or features["has_table_request"]
        or features["has_json_request"]
        or token_count > 140
    ):
        return ClassificationResult(
            "moderate",
            "Summarization, comparison, structured analysis, or medium-length task.",
            "rules",
        )

    return ClassificationResult(
        "moderate",
        "Defaulted to moderate because the task is not clearly simple or complex.",
        "rules",
    )

