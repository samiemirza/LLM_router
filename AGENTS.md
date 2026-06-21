# AGENTS.md

## Project Name

LLM Cost Autopilot

## Project Goal

Build a clean, understandable MVP of an LLM routing layer that reduces API cost by routing prompts to the cheapest model capable of handling the task, while logging cost, latency, routing decisions, and quality verification results.

The project should be portfolio-ready for AI Engineering roles. Prioritize clarity, modularity, observability, and reproducibility over cleverness.

## Core Product Behavior

A user sends a chat completion request to the API.

The system should:

1. Extract basic features from the prompt.
2. Predict task complexity: `simple`, `moderate`, or `complex`.
3. Select a model based on configurable routing rules.
4. Call the selected provider.
5. Estimate cost from token usage and configured pricing.
6. Log the request, routing decision, cost, latency, and response metadata.
7. Optionally verify quality using a stronger model or deterministic checks.
8. Expose metrics through a Streamlit dashboard.

## MVP Constraints

Use Python 3.11+, FastAPI, Streamlit, SQLite, scikit-learn, YAML config, and the OpenAI Python SDK for OpenAI-compatible providers.

Do not add Kubernetes, PostgreSQL, user accounts, frontend frameworks, complex authentication, background queues, fine-tuned models, or unnecessary abstractions.

## Implementation Principles

Model IDs, providers, quality tiers, and prices must come from `config/models.yaml`.

Routing rules, baseline model, and verification settings must come from `config/routing.yaml`.

Provider-specific logic belongs in `app/providers/`.

Routing logic belongs in `app/router.py`.

Cost calculation belongs in `app/cost.py`.

Database logic belongs in `app/logging_db.py`.

Classifier logic belongs in `app/classifier.py`.

Never hard-code API keys. Never commit real `.env` values.

When `LLM_MOCK_MODE=true`, the app must avoid paid API calls and return deterministic fake model responses for local testing.

