"""Prompt feature extraction for routing and classifier training."""

from __future__ import annotations

import math
import re
from typing import Any


FEATURE_NAMES = [
    "char_length",
    "approx_token_count",
    "message_count",
    "has_json_request",
    "has_table_request",
    "has_bullet_request",
    "has_code",
    "has_reasoning_keywords",
    "has_comparison_keywords",
    "has_simple_task_keywords",
    "estimated_constraint_count",
]

SIMPLE_TASK_KEYWORDS = [
    "rewrite",
    "rephrase",
    "fix grammar",
    "grammar",
    "extract",
    "convert",
    "format",
    "return only",
    "make this shorter",
    "shorten",
    "cleanup",
    "clean up",
    "title case",
    "polite",
]

REASONING_KEYWORDS = [
    "analyze",
    "reason",
    "why",
    "step by step",
    "tradeoff",
    "trade-off",
    "root cause",
    "debug",
    "architecture",
    "strategy",
    "optimize",
    "design",
    "evaluate",
    "justify",
]

COMPARISON_KEYWORDS = [
    "compare",
    "versus",
    "vs",
    "pros and cons",
    "advantages",
    "disadvantages",
    "better option",
    "benchmark",
]

CODE_PATTERNS = [
    r"```",
    r"\bdef\s+\w+\(",
    r"\bclass\s+\w+",
    r"\bimport\s+\w+",
    r"\bfunction\s+\w+\(",
    r"console\.log",
    r"Traceback \(most recent call last\)",
]


def get_prompt_text(messages: list[dict[str, Any]]) -> str:
    """Join chat messages into one prompt string for analysis."""
    return "\n".join(str(message.get("content", "")) for message in messages)


def approximate_token_count(text: str) -> int:
    """Cheap token estimate good enough for routing and mock mode."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _has_code(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in CODE_PATTERNS)


def _estimate_constraint_count(text: str) -> int:
    marker_count = len(
        re.findall(
            r"\b(must|only|include|exclude|without|with|exactly|at least|no more than|format|return)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    comma_count = text.count(",")
    bullet_count = len(re.findall(r"(^|\n)\s*[-*]\s+", text))
    numbered_count = len(re.findall(r"(^|\n)\s*\d+[.)]\s+", text))
    return marker_count + comma_count + bullet_count + numbered_count


def extract_features(messages: list[dict[str, Any]]) -> dict[str, int]:
    """Extract simple, interpretable prompt features."""
    prompt = get_prompt_text(messages)
    lowered = prompt.lower()

    features = {
        "char_length": len(prompt),
        "approx_token_count": approximate_token_count(prompt),
        "message_count": len(messages),
        "has_json_request": int("json" in lowered or "schema" in lowered),
        "has_table_request": int(
            "table" in lowered or "spreadsheet" in lowered or "csv" in lowered
        ),
        "has_bullet_request": int(
            "bullet" in lowered or "bullets" in lowered or "bullet points" in lowered
        ),
        "has_code": int(_has_code(prompt)),
        "has_reasoning_keywords": int(_contains_any(lowered, REASONING_KEYWORDS)),
        "has_comparison_keywords": int(_contains_any(lowered, COMPARISON_KEYWORDS)),
        "has_simple_task_keywords": int(_contains_any(lowered, SIMPLE_TASK_KEYWORDS)),
        "estimated_constraint_count": _estimate_constraint_count(prompt),
    }
    return features


def features_to_vector(features: dict[str, int]) -> list[int]:
    """Convert named features to a stable vector for scikit-learn."""
    return [int(features.get(name, 0)) for name in FEATURE_NAMES]

