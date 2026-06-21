"""SQLite logging and metrics helpers."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.config_loader import get_database_path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path or get_database_path())
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: str | None = None) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                prompt_hash TEXT,
                prompt_preview TEXT,
                complexity_tier TEXT,
                selected_model_key TEXT,
                selected_model_id TEXT,
                provider TEXT,
                routing_reason TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL,
                latency_ms REAL,
                response_text TEXT,
                baseline_model_key TEXT,
                baseline_estimated_cost REAL,
                estimated_savings REAL,
                estimated_savings_percent REAL,
                verification_status TEXT,
                error_message TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER,
                verifier_model_key TEXT,
                quality_score REAL,
                passed INTEGER,
                failure_reason TEXT,
                verifier_cost REAL,
                created_at TEXT
            )
            """
        )


def log_request(
    *,
    prompt: str,
    complexity_tier: str,
    selected_model_key: str,
    selected_model_id: str,
    provider: str,
    routing_reason: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    latency_ms: float,
    response_text: str,
    baseline_model_key: str,
    baseline_estimated_cost: float,
    estimated_savings: float,
    estimated_savings_percent: float,
    verification_status: str,
    error_message: str | None = None,
    db_path: str | None = None,
) -> int:
    init_db(db_path)
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO requests (
                timestamp,
                prompt_hash,
                prompt_preview,
                complexity_tier,
                selected_model_key,
                selected_model_id,
                provider,
                routing_reason,
                input_tokens,
                output_tokens,
                cost,
                latency_ms,
                response_text,
                baseline_model_key,
                baseline_estimated_cost,
                estimated_savings,
                estimated_savings_percent,
                verification_status,
                error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                hash_prompt(prompt),
                prompt[:250],
                complexity_tier,
                selected_model_key,
                selected_model_id,
                provider,
                routing_reason,
                input_tokens,
                output_tokens,
                cost,
                latency_ms,
                response_text,
                baseline_model_key,
                baseline_estimated_cost,
                estimated_savings,
                estimated_savings_percent,
                verification_status,
                error_message,
            ),
        )
        return int(cursor.lastrowid)


def log_verification(
    *,
    request_id: int | None,
    verifier_model_key: str | None,
    quality_score: float,
    passed: bool,
    failure_reason: str | None,
    verifier_cost: float,
    db_path: str | None = None,
) -> int:
    init_db(db_path)
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO verifications (
                request_id,
                verifier_model_key,
                quality_score,
                passed,
                failure_reason,
                verifier_cost,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                verifier_model_key,
                quality_score,
                int(passed),
                failure_reason,
                verifier_cost,
                utc_now_iso(),
            ),
        )
        return int(cursor.lastrowid)


def get_stats(db_path: str | None = None) -> dict[str, Any]:
    init_db(db_path)
    with get_connection(db_path) as connection:
        totals = connection.execute(
            """
            SELECT
                COUNT(*) AS total_requests,
                COALESCE(SUM(cost), 0) AS total_actual_cost,
                COALESCE(SUM(baseline_estimated_cost), 0) AS total_baseline_cost,
                COALESCE(SUM(estimated_savings), 0) AS total_estimated_savings,
                COALESCE(AVG(latency_ms), 0) AS average_latency_ms
            FROM requests
            """
        ).fetchone()

        by_model = connection.execute(
            """
            SELECT selected_model_key, COUNT(*) AS count
            FROM requests
            GROUP BY selected_model_key
            ORDER BY count DESC
            """
        ).fetchall()

        by_complexity = connection.execute(
            """
            SELECT complexity_tier, COUNT(*) AS count
            FROM requests
            GROUP BY complexity_tier
            ORDER BY count DESC
            """
        ).fetchall()

        quality = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(passed), 0) AS passed
            FROM verifications
            """
        ).fetchone()

    total_baseline = float(totals["total_baseline_cost"])
    total_savings = float(totals["total_estimated_savings"])
    savings_percentage = round((total_savings / total_baseline) * 100, 2) if total_baseline else 0.0
    quality_total = int(quality["total"])
    quality_pass_rate = (
        round((int(quality["passed"]) / quality_total) * 100, 2) if quality_total else None
    )

    return {
        "total_requests": int(totals["total_requests"]),
        "total_actual_cost": round(float(totals["total_actual_cost"]), 10),
        "total_baseline_cost": round(total_baseline, 10),
        "total_estimated_savings": round(total_savings, 10),
        "savings_percentage": savings_percentage,
        "average_latency_ms": round(float(totals["average_latency_ms"]), 2),
        "requests_by_model": {row["selected_model_key"]: row["count"] for row in by_model},
        "requests_by_complexity": {row["complexity_tier"]: row["count"] for row in by_complexity},
        "quality_pass_rate": quality_pass_rate,
    }


def get_recent_requests(limit: int = 50, db_path: str | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM requests
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_requests_dataframe(db_path: str | None = None) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as connection:
        return pd.read_sql_query("SELECT * FROM requests ORDER BY id DESC", connection)

