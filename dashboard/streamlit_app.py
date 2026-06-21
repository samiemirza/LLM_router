"""Streamlit dashboard for LLM Cost Autopilot."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.logging_db import get_all_requests_dataframe, get_stats  # noqa: E402


st.set_page_config(page_title="LLM Cost Autopilot Dashboard", layout="wide")
st.title("LLM Cost Autopilot Dashboard")

stats = get_stats()
requests_df = get_all_requests_dataframe()

metric_cols = st.columns(6)
metric_cols[0].metric("Total Requests", stats["total_requests"])
metric_cols[1].metric("Actual Cost", f"${stats['total_actual_cost']:.6f}")
metric_cols[2].metric("Baseline Cost", f"${stats['total_baseline_cost']:.6f}")
metric_cols[3].metric("Savings", f"${stats['total_estimated_savings']:.6f}")
metric_cols[4].metric("Savings %", f"{stats['savings_percentage']:.2f}%")
metric_cols[5].metric("Avg Latency", f"{stats['average_latency_ms']:.0f} ms")

if requests_df.empty:
    st.info("No requests logged yet. Run the API or benchmark in mock mode to populate the dashboard.")
    st.stop()

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Requests by Model")
    model_counts = requests_df["selected_model_key"].value_counts()
    st.bar_chart(model_counts)

with right_col:
    st.subheader("Requests by Complexity")
    complexity_counts = requests_df["complexity_tier"].value_counts()
    st.bar_chart(complexity_counts)

st.subheader("Cost Over Time")
requests_df["timestamp"] = pd.to_datetime(requests_df["timestamp"], errors="coerce")
cost_over_time = (
    requests_df.dropna(subset=["timestamp"])
    .sort_values("timestamp")
    .set_index("timestamp")[["cost", "baseline_estimated_cost", "estimated_savings"]]
)
if len(cost_over_time) >= 2:
    st.line_chart(cost_over_time)
else:
    st.caption("Cost trend will appear after at least two logged requests.")

st.subheader("Recent Requests")
columns = [
    "id",
    "timestamp",
    "complexity_tier",
    "selected_model_key",
    "cost",
    "baseline_estimated_cost",
    "estimated_savings",
    "latency_ms",
    "verification_status",
    "prompt_preview",
]
st.dataframe(requests_df[columns].head(50), use_container_width=True)

