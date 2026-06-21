"""Train a scikit-learn complexity classifier."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.features import FEATURE_NAMES, extract_features, features_to_vector  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "labeled_prompts.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "complexity_classifier.pkl"
METRICS_PATH = PROJECT_ROOT / "reports" / "classifier_metrics.json"
LABELS = ["simple", "moderate", "complex"]


def load_rows() -> tuple[list[list[int]], list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run python scripts/seed_labeled_prompts.py first.")

    vectors: list[list[int]] = []
    labels: list[str] = []
    with DATA_PATH.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            messages = [{"role": "user", "content": row["prompt"]}]
            features = extract_features(messages)
            vectors.append(features_to_vector(features))
            labels.append(row["label"])
    return vectors, labels


def main() -> None:
    vectors, labels = load_rows()
    x_train, x_test, y_train, y_test = train_test_split(
        vectors,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "labels": LABELS,
        "feature_names": FEATURE_NAMES,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=LABELS).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=LABELS,
            zero_division=0,
            output_dict=True,
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump({"model": model, "feature_names": FEATURE_NAMES}, file)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Saved classifier to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()

