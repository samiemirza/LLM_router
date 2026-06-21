"""Generate starter benchmark prompts."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval_prompts.jsonl"


TASKS = [
    ("Rewrite this message professionally: Need the file today.", "simple", "rewrite"),
    ("Fix grammar only: He dont have the final version.", "simple", "grammar"),
    ("Extract the email address from: Contact maria@company.com for invoice help.", "simple", "extraction"),
    ("Extract the phone number from this text: Call 03001234567 before 5 PM.", "simple", "extraction"),
    ("Convert this sentence into a polite request: Send me the document.", "simple", "rewrite"),
    ("Turn these skills into a comma-separated list: Python SQL FastAPI Docker.", "simple", "formatting"),
    ("Return this as JSON with name and role: Ali Khan Data Analyst.", "simple", "json_formatting"),
    ("Make this sentence shorter without changing meaning: I am writing to ask for updates.", "simple", "rewrite"),
    ("Fix capitalization: i worked at systems limited as an ai intern.", "simple", "formatting"),
    ("Extract the date from: The deadline is 30 June 2026.", "simple", "extraction"),
    ("Convert this paragraph into three bullet points: The app tracks costs, routes prompts, and logs results.", "simple", "formatting"),
    ("Rewrite this Slack message to sound less direct: You forgot to send the update.", "simple", "rewrite"),
    ("Return only the company name: I interviewed with Intellia for a summer internship.", "simple", "extraction"),
    ("Format this address neatly: house 12 street 4 dha phase 5 karachi.", "simple", "formatting"),
    ("Extract the amount from this sentence: The total cost was $42.75.", "simple", "extraction"),
    ("Convert this casual line into a formal sentence: Hey can you review my resume?", "simple", "rewrite"),
    ("Remove duplicate words: This model model is very very useful.", "simple", "cleanup"),
    ("Return only the city mentioned here: The event will be held in Islamabad.", "simple", "extraction"),
    ("Fix spelling: I am intrested in artificail inteligence and machne learning.", "simple", "grammar"),
    ("Turn this into a resume bullet: I built a dashboard that showed API costs.", "simple", "resume_rewrite"),
    ("Summarize this project description into five bullets for a recruiter.", "moderate", "summarization"),
    ("Compare FastAPI and Flask for a lightweight ML API.", "moderate", "comparison"),
    ("Classify these messages by urgency and topic: payment failed, app crashed, password reset.", "moderate", "classification"),
    ("Write a customer update email explaining a one-week delay.", "moderate", "business_writing"),
    ("Create a table comparing SQLite, PostgreSQL, and MySQL for an MVP.", "moderate", "structured_analysis"),
    ("Analyze these user comments and identify product improvement themes.", "moderate", "analysis"),
    ("Turn these rough notes into a clear implementation plan for a dashboard.", "moderate", "planning"),
    ("Draft a short proposal for an LLM cost monitoring tool.", "moderate", "business_writing"),
    ("Summarize this meeting into decisions, blockers, and action items.", "moderate", "summarization"),
    ("Recommend which metrics an LLM routing dashboard should show.", "moderate", "recommendation"),
    ("Create acceptance criteria for a /v1/stats API endpoint.", "moderate", "requirements"),
    ("Compare rules-based routing and classifier-based routing in a short table.", "moderate", "comparison"),
    ("Write a README section explaining mock mode for local testing.", "moderate", "documentation"),
    ("Analyze why a cheap model might fail on long-context prompts.", "moderate", "analysis"),
    ("Categorize ten example prompts into simple, moderate, and complex tiers.", "moderate", "classification"),
    ("Create a JSON schema for logging request metadata.", "moderate", "json_schema"),
    ("Summarize an architecture diagram in business-friendly language.", "moderate", "summarization"),
    ("Write a concise risk section for using third-party LLM APIs.", "moderate", "business_writing"),
    ("Compare cost savings and latency savings for model routing.", "moderate", "comparison"),
    ("Turn these roadmap ideas into prioritized milestones.", "moderate", "planning"),
    ("Debug this Python error and explain the root cause: TypeError object is not subscriptable.", "complex", "debugging"),
    ("Design an architecture for an LLM router with provider fallback and observability.", "complex", "architecture"),
    ("Create a strategy for reducing LLM API spend while preserving answer quality.", "complex", "strategy"),
    ("Analyze failure modes for LLM-as-judge verification in production.", "complex", "analysis"),
    ("Explain step by step how to test provider integrations without paid API calls.", "complex", "testing_strategy"),
    ("Design a schema for logging prompts safely without storing sensitive data.", "complex", "privacy_design"),
    ("Evaluate tradeoffs between routing by rules and a trained classifier.", "complex", "nuanced_judgment"),
    ("Plan a V2 architecture with async verification and historical quality metrics.", "complex", "architecture"),
    ("Debug why a trained classifier might route too many prompts to complex models.", "complex", "debugging"),
    ("Design a fallback policy for OpenAI and OpenRouter outages.", "complex", "resilience"),
    ("Analyze cost, latency, and quality tradeoffs for a multi-provider LLM gateway.", "complex", "analysis"),
    ("Create an implementation plan for adding per-team cost attribution.", "complex", "architecture"),
    ("Explain how to handle long-context prompts with strict cost controls.", "complex", "strategy"),
    ("Design deterministic verification checks for JSON, bullets, and instruction following.", "complex", "verification"),
    ("Analyze the security risks of exposing an OpenAI-compatible routing endpoint.", "complex", "security"),
    ("Create a detailed rollout plan for migrating an app from direct LLM calls to a router.", "complex", "rollout"),
    ("Optimize a request logging design for reliability under concurrent traffic.", "complex", "systems_design"),
    ("Evaluate whether prompt hashing is sufficient for privacy-preserving analytics.", "complex", "privacy_analysis"),
    ("Design a benchmark methodology for measuring routing savings against a baseline.", "complex", "evaluation"),
    ("Create a nuanced escalation policy for when a cheap model response should be retried.", "complex", "policy"),
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for index, (prompt, expected_tier, task_type) in enumerate(TASKS, start=1):
            row = {
                "id": f"eval_{index:03d}",
                "prompt": prompt,
                "expected_tier": expected_tier,
                "task_type": task_type,
            }
            file.write(json.dumps(row) + "\n")
    print(f"Wrote {len(TASKS)} eval prompts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

