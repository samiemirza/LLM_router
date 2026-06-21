"""Generate starter labeled prompts for classifier training."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "labeled_prompts.csv"


SIMPLE_PROMPTS = [
    "Rewrite this email to sound more professional: I need the report by tonight.",
    "Fix the grammar in this sentence: She dont know where the files is.",
    "Convert this paragraph into three bullet points.",
    "Extract the email address from this text: Contact Ahmed at ahmed@example.com.",
    "Extract the phone number from this message: Please call 03001234567 after lunch.",
    "Rewrite this message in a polite tone: Send me the update now.",
    "Make this sentence shorter: The reason I am writing is because I wanted to ask.",
    "Convert this list into a comma-separated format: apples bananas oranges mangoes.",
    "Return the following name and email as JSON: Sara Khan sara.khan@example.com.",
    "Fix capitalization in this sentence: i am applying for the data analyst role.",
    "Rewrite this LinkedIn message to sound friendly and concise.",
    "Extract the company name from this sentence: I worked at Systems Limited.",
    "Convert this text into title case: introduction to machine learning pipelines.",
    "Remove duplicate words from this sentence: This is is very very simple.",
    "Summarize this one sentence into fewer words.",
    "Format this address neatly: house 12 street 5 dha phase 6 karachi.",
    "Turn this casual message into a formal email sentence: Hey can you check my CV.",
    "Extract the date from this sentence: The interview is scheduled for 25 June 2026.",
    "Convert this sentence into past tense: I am working on the dashboard.",
    "Return only the city mentioned here: The event is in Islamabad.",
    "Clean up this sentence without changing the meaning.",
    "Rewrite this complaint so it sounds calm and professional.",
    "Extract the invoice number from this message: Invoice INV-2048 is overdue.",
    "Convert these words into a numbered list: cost latency quality routing.",
    "Fix spelling mistakes in this line.",
    "Return a JSON object with name, role, and company from this text.",
    "Make this Slack message less direct: You forgot the attachment.",
    "Extract the amount from this sentence: The total cost was $42.75.",
    "Turn this sentence into a resume bullet.",
    "Format this meeting time clearly: fri 3pm pakistan time.",
]


MODERATE_PROMPTS = [
    "Summarize this product update into an executive brief with risks and next steps.",
    "Compare FastAPI and Flask for a small internal ML service.",
    "Classify these customer tickets into billing, technical, and account categories.",
    "Write a concise business email explaining a delayed project timeline.",
    "Create a table comparing three cloud storage options for a startup.",
    "Analyze this user feedback and identify the top three product themes.",
    "Summarize this meeting transcript into decisions, owners, and deadlines.",
    "Draft a project proposal for an AI support chatbot MVP.",
    "Turn these notes into a structured implementation plan.",
    "Categorize these prompts by likely complexity tier.",
    "Compare two pricing strategies for an API product.",
    "Write a short recommendation for which database to use in an MVP.",
    "Summarize a technical blog post for a non-technical stakeholder.",
    "Create a JSON schema for a simple customer profile object.",
    "Explain the tradeoffs between SQLite and PostgreSQL for a portfolio project.",
    "Review this product spec and list missing requirements.",
    "Build a markdown table of model options with cost and quality columns.",
    "Write a professional response to a customer asking for a refund.",
    "Summarize key metrics from this dashboard description.",
    "Classify these support messages by urgency and topic.",
    "Create interview talking points for an AI engineering project.",
    "Compare batch evaluation and online evaluation for LLM routing.",
    "Write acceptance criteria for a cost tracking API endpoint.",
    "Analyze these logs and identify likely operational issues.",
    "Create a short onboarding guide for a new API user.",
    "Summarize this policy document into practical action items.",
    "Draft release notes for a new dashboard feature.",
    "Compare the risks of using a small model versus a large model.",
    "Turn this rough project outline into a clear README section.",
    "Recommend a simple evaluation plan for an LLM router MVP.",
]


COMPLEX_PROMPTS = [
    "Debug this Python traceback and explain the root cause plus a safe fix.",
    "Design the architecture for a multi-tenant LLM gateway with audit logging.",
    "Create a strategy for reducing LLM cost without hurting answer quality.",
    "Analyze this incident timeline and identify systemic failures and mitigations.",
    "Refactor this codebase plan to improve maintainability and testability.",
    "Design an evaluation framework for routing prompts across multiple LLM providers.",
    "Explain step by step how to migrate a synchronous API to async safely.",
    "Build a nuanced argument for whether to use LLM-as-judge in production.",
    "Optimize this SQL query and explain why the current plan is slow.",
    "Design a fallback system for provider outages and partial API failures.",
    "Analyze security risks in an OpenAI-compatible proxy service.",
    "Create an architecture decision record for choosing SQLite in an MVP.",
    "Debug a failing Docker Compose setup with API and dashboard services.",
    "Design prompt routing rules for conflicting cost, latency, and quality goals.",
    "Evaluate the tradeoffs of classifier-driven routing versus rules-only routing.",
    "Write a detailed implementation plan for observability in an LLM platform.",
    "Analyze model pricing changes and propose a robust configuration strategy.",
    "Design a test strategy for avoiding paid API calls in CI.",
    "Create a failure-mode analysis for LLM provider integrations.",
    "Explain how to handle long-context routing with strict cost controls.",
    "Debug why a trained classifier is underperforming on complex prompts.",
    "Design a schema for logging prompt metadata without storing sensitive data.",
    "Analyze this architecture and identify bottlenecks, risks, and alternatives.",
    "Plan a V2 roadmap for a production-grade LLM routing platform.",
    "Create a nuanced policy for when to escalate from a cheap model to a strong model.",
    "Design deterministic and LLM-based verification for structured outputs.",
    "Debug a race condition in request logging under concurrent FastAPI traffic.",
    "Evaluate whether prompt hashing is enough for privacy and observability.",
    "Design a cost attribution system for teams using shared API keys.",
    "Create a detailed rollout strategy for replacing direct LLM calls with a router.",
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        [(prompt, "simple") for prompt in SIMPLE_PROMPTS]
        + [(prompt, "moderate") for prompt in MODERATE_PROMPTS]
        + [(prompt, "complex") for prompt in COMPLEX_PROMPTS]
    )
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} labeled prompts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

