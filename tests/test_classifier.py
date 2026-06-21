from app.classifier import classify_with_rules
from app.features import extract_features


def classify_prompt(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    result = classify_with_rules(extract_features(messages), prompt)
    return result.complexity_tier


def test_rule_classifier_simple_rewrite() -> None:
    assert classify_prompt("Rewrite this professionally: send me the report.") == "simple"


def test_rule_classifier_moderate_summary() -> None:
    prompt = "Summarize this project description into risks, milestones, and next steps."
    assert classify_prompt(prompt) == "moderate"


def test_rule_classifier_complex_architecture() -> None:
    prompt = "Design the architecture for an LLM routing gateway with fallback and observability."
    assert classify_prompt(prompt) == "complex"

