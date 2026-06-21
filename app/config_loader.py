"""Load YAML configuration and environment-derived settings."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
ROUTING_CONFIG_PATH = PROJECT_ROOT / "config" / "routing.yaml"

_runtime_routing_config: dict[str, Any] | None = None


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def load_models_config(path: Path = MODELS_CONFIG_PATH) -> dict[str, Any]:
    data = load_yaml_file(path)
    models = data.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("config/models.yaml must contain a non-empty 'models' mapping.")
    return models


def load_routing_config(path: Path = ROUTING_CONFIG_PATH) -> dict[str, Any]:
    if _runtime_routing_config is not None:
        return copy.deepcopy(_runtime_routing_config)
    return load_yaml_file(path)


def update_runtime_routing_config(new_config: dict[str, Any]) -> dict[str, Any]:
    """Merge a runtime routing update for the current process only."""
    global _runtime_routing_config
    base_config = load_routing_config()
    _deep_update(base_config, new_config)
    _runtime_routing_config = base_config
    return copy.deepcopy(_runtime_routing_config)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def is_mock_mode() -> bool:
    return env_bool("LLM_MOCK_MODE", default=False)


def get_database_path(database_url: str | None = None) -> str:
    """Resolve sqlite:/// URLs to a path sqlite3 can open."""
    database_url = database_url or os.getenv("DATABASE_URL", "sqlite:///./llm_cost_autopilot.db")
    if database_url == "sqlite:///:memory:":
        return ":memory:"
    if not database_url.startswith("sqlite:///"):
        raise ValueError("Only sqlite:/// DATABASE_URL values are supported in this MVP.")

    raw_path = database_url.removeprefix("sqlite:///")
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_app_name() -> str:
    return os.getenv("APP_NAME", "LLM Cost Autopilot")


def get_app_url() -> str:
    return os.getenv("APP_URL", "http://localhost:8000")

