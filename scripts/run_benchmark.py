"""Run benchmark prompts through the router."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_PATH = PROJECT_ROOT / "data" / "eval_prompts.jsonl"
RESULTS_PATH = PROJECT_ROOT / "reports" / "benchmark_results.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "benchmark_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM Cost Autopilot benchmark prompts.")
    parser.add_argument("--mock", action="store_true", help="Force LLM_MOCK_MODE=true.")
    return parser.parse_args()


def load_prompts() -> list[dict[str, Any]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run python scripts/seed_eval_prompts.py first.")

    rows = []
    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def main() -> None:
    args = parse_args()
    if args.mock:
        os.environ["LLM_MOCK_MODE"] = "true"

    from app.main import process_chat_completion  # noqa: WPS433
    from app.schemas import ChatCompletionRequest, ChatMessage  # noqa: WPS433

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = load_prompts()
    results: list[dict[str, Any]] = []

    for row in rows:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=row["prompt"])],
            temperature=0.2,
            max_tokens=500,
        )
        response = process_chat_completion(request)
        metadata = model_dump(response.metadata)
        results.append(
            {
                "id": row["id"],
                "task_type": row["task_type"],
                "expected_tier": row["expected_tier"],
                "predicted_tier": metadata["complexity_tier"],
                "selected_model_key": metadata["selected_model_key"],
                "provider": metadata["provider"],
                "input_tokens": metadata["input_tokens"],
                "output_tokens": metadata["output_tokens"],
                "actual_cost": metadata["cost"],
                "baseline_cost": metadata["baseline_estimated_cost"],
                "estimated_savings": metadata["estimated_savings"],
                "estimated_savings_percent": metadata["estimated_savings_percent"],
                "latency_ms": metadata["latency_ms"],
                "verification_status": metadata["verification_status"],
            }
        )

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    total_actual_cost = sum(float(row["actual_cost"]) for row in results)
    total_baseline_cost = sum(float(row["baseline_cost"]) for row in results)
    estimated_savings = total_baseline_cost - total_actual_cost
    savings_percent = (
        estimated_savings / total_baseline_cost * 100 if total_baseline_cost else 0.0
    )
    summary = {
        "total_prompts": len(results),
        "total_actual_cost": round(total_actual_cost, 10),
        "total_baseline_cost": round(total_baseline_cost, 10),
        "estimated_savings": round(estimated_savings, 10),
        "estimated_savings_percent": round(savings_percent, 2),
        "average_latency_ms": round(
            sum(float(row["latency_ms"]) for row in results) / len(results),
            2,
        ),
        "routing_distribution": dict(Counter(row["selected_model_key"] for row in results)),
        "complexity_distribution": dict(Counter(row["predicted_tier"] for row in results)),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Wrote benchmark results to {RESULTS_PATH}")
    print(f"Wrote benchmark summary to {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

