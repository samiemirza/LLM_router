from app.logging_db import get_recent_requests, get_stats, init_db, log_request, log_verification


def test_sqlite_logging_round_trip(tmp_path) -> None:
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    request_id = log_request(
        prompt="Rewrite this professionally.",
        complexity_tier="simple",
        selected_model_key="cheap_model",
        selected_model_id="provider/model",
        provider="openrouter",
        routing_reason="Short rewrite task.",
        input_tokens=10,
        output_tokens=5,
        cost=0.001,
        latency_ms=12.5,
        response_text="Mock response",
        baseline_model_key="strong_model",
        baseline_estimated_cost=0.01,
        estimated_savings=0.009,
        estimated_savings_percent=90.0,
        verification_status="not_sampled",
        db_path=db_path,
    )
    log_verification(
        request_id=request_id,
        verifier_model_key=None,
        quality_score=5.0,
        passed=True,
        failure_reason=None,
        verifier_cost=0.0,
        db_path=db_path,
    )

    recent = get_recent_requests(db_path=db_path)
    stats = get_stats(db_path=db_path)

    assert recent[0]["id"] == request_id
    assert stats["total_requests"] == 1
    assert stats["total_actual_cost"] == 0.001
    assert stats["quality_pass_rate"] == 100.0

