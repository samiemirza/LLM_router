# LLM Cost Autopilot

LLM Cost Autopilot routes prompts by complexity to reduce LLM inference cost while tracking quality, latency, and estimated savings against a high-tier baseline.

## Architecture Overview

The API accepts OpenAI-style chat requests, extracts prompt features, classifies task complexity, routes the request to a configured provider/model, calculates cost from token usage, logs the decision to SQLite, and exposes metrics through a Streamlit dashboard.

```text
Client
  |
  v
FastAPI /v1/completions
  |
  +--> app/features.py      -> prompt feature extraction
  +--> app/classifier.py    -> rules or trained sklearn classifier
  +--> app/router.py        -> config-driven model selection
  +--> app/providers/       -> OpenAI, OpenRouter, or mock provider
  +--> app/cost.py          -> actual and baseline cost estimates
  +--> app/verifier.py      -> deterministic or optional LLM quality checks
  +--> app/logging_db.py    -> SQLite request and verification logs
  |
  v
Streamlit Dashboard -> reads SQLite metrics and recent requests
```

## Features

- Config-driven model IDs, providers, quality tiers, and prices in `config/models.yaml`
- Config-driven routing, baseline, and verification settings in `config/routing.yaml`
- Mock mode for local development without paid API calls
- Rule-based fallback classifier plus optional trained scikit-learn classifier
- SQLite logging for requests, routing decisions, costs, latency, and verification metadata
- Streamlit dashboard for cost, savings, latency, routing distribution, and recent requests
- Benchmark runner that compares routed cost against a configured baseline model

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

The default `.env.example` uses mock mode, so the app runs without real API keys:

```env
OPENAI_API_KEY=your_openai_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DATABASE_URL=sqlite:///./llm_cost_autopilot.db
LLM_MOCK_MODE=true
VERIFICATION_SAMPLE_RATE=0.20
APP_NAME=LLM Cost Autopilot
APP_URL=http://localhost:8000
```

For real provider calls, set valid keys and change `LLM_MOCK_MODE=false`.

## Run API

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## Run Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard reads the same SQLite database configured by `DATABASE_URL`.

## Train Classifier

```bash
python scripts/seed_labeled_prompts.py
python scripts/train_classifier.py
```

Training writes:

- `models/complexity_classifier.pkl`
- `reports/classifier_metrics.json`

If the classifier file does not exist, the API automatically uses transparent rule-based routing.

## Run Benchmark

```bash
python scripts/seed_eval_prompts.py
python scripts/run_benchmark.py --mock
```

Benchmark outputs:

- `reports/benchmark_results.csv`
- `reports/benchmark_summary.json`

## Example Request

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Rewrite this professionally: send me the report today."}
    ],
    "temperature": 0.2,
    "max_tokens": 500
  }'
```

## Example Response

```json
{
  "response": "Mock response from DeepSeek V4 Flash: Rewrite this professionally: send me the report today.",
  "metadata": {
    "request_id": 1,
    "selected_model_key": "deepseek_v4_flash",
    "selected_model_id": "deepseek/deepseek-v4-flash",
    "provider": "openrouter",
    "complexity_tier": "simple",
    "routing_reason": "Short rewriting, extraction, cleanup, or formatting task.",
    "input_tokens": 14,
    "output_tokens": 21,
    "cost": 0.000005,
    "latency_ms": 0.03,
    "baseline_model_key": "gpt_5_5",
    "baseline_estimated_cost": 0.0007,
    "estimated_savings": 0.000695,
    "estimated_savings_percent": 99.29,
    "verification_queued_or_sampled": false,
    "verification_status": "not_sampled"
  }
}
```

## Cost Savings Formula

Actual cost:

```python
cost = (
    input_tokens / 1_000_000 * input_cost_per_1m
    + output_tokens / 1_000_000 * output_cost_per_1m
)
```

Estimated savings:

```python
estimated_savings = baseline_estimated_cost - actual_cost
estimated_savings_percent = estimated_savings / baseline_estimated_cost * 100
```

The baseline model is configured in `config/routing.yaml`.

## API Endpoints

- `GET /health`
- `POST /v1/completions`
- `GET /v1/models`
- `GET /v1/stats`
- `GET /v1/recent-requests`
- `PUT /v1/routing-config`
- `POST /v1/evaluate`

## Docker

```bash
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

## Tests

```bash
pytest
```

Tests use mock mode patterns and do not call paid APIs.

## MVP Limitations

- Token counts are provider-reported for real calls and approximate in mock mode.
- Async/background verification is represented by sampled status metadata, not a queue.
- The trained classifier uses a small synthetic starter dataset.
- Runtime routing updates are in-memory for the current process only.
- OpenRouter attribution headers are included through the OpenAI SDK when supported.

## V2 Roadmap

- Async verification worker and scheduled quality audits
- Provider retry policies with richer error classification
- Historical model price tracking
- Per-project or per-team cost attribution
- Prompt privacy controls and configurable redaction
- Better classifier datasets and active-learning feedback loops
- Optional OpenAI-compatible proxy endpoint for drop-in client migration

