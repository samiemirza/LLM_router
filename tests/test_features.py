from app.features import extract_features


def test_extract_features_detects_structured_requests() -> None:
    messages = [
        {
            "role": "user",
            "content": "Return JSON with name and email, then convert it into bullet points.",
        }
    ]

    features = extract_features(messages)

    assert features["has_json_request"] == 1
    assert features["has_bullet_request"] == 1
    assert features["estimated_constraint_count"] >= 2


def test_extract_features_detects_code() -> None:
    messages = [{"role": "user", "content": "Debug this:\n```python\ndef x(): pass\n```"}]

    features = extract_features(messages)

    assert features["has_code"] == 1
    assert features["has_reasoning_keywords"] == 1

