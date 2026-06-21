"""Export a dashboard-safe snapshot for Streamlit Cloud demos."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.logging_db import get_all_requests_dataframe  # noqa: E402


OUTPUT_PATH = PROJECT_ROOT / "data" / "dashboard_snapshot.csv"
SNAPSHOT_COLUMNS = [
    "id",
    "timestamp",
    "prompt_preview",
    "complexity_tier",
    "selected_model_key",
    "selected_model_id",
    "provider",
    "routing_reason",
    "input_tokens",
    "output_tokens",
    "cost",
    "latency_ms",
    "baseline_model_key",
    "baseline_estimated_cost",
    "estimated_savings",
    "estimated_savings_percent",
    "verification_status",
    "error_message",
]


def main() -> None:
    requests_df = get_all_requests_dataframe()
    if requests_df.empty:
        raise SystemExit("No request rows found. Run the benchmark or send API requests first.")

    snapshot_df = requests_df[SNAPSHOT_COLUMNS].copy()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(snapshot_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

