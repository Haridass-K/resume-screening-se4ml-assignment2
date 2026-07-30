from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "Resume.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "production" / "resume_screening_model.pkl"

OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "production" / "model_quality_metrics.csv"


def calculate_model_quality_metrics():
    """Calculate classification metrics for the production model."""

    data = pd.read_csv(DATA_PATH)

    X = data["Resume_str"]
    y = data["Category"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Weighted Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "Weighted Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "Weighted F1-Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    results = pd.DataFrame(
        {
            "Metric": metrics.keys(),
            "Value": metrics.values(),
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nModel Quality Metrics")
    print("-" * 40)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f} ({value * 100:.2f}%)")

    print(f"\nResults saved to: {OUTPUT_PATH}")


def calculate_data_quality_metrics():
    """Calculate quality metrics for the resume dataset."""

    data = pd.read_csv(DATA_PATH)

    required_columns = ["Resume_str", "Category"]
    total_rows = len(data)

    # 1. Missing value rate across required columns
    missing_values = data[required_columns].isna().sum().sum()
    total_required_values = total_rows * len(required_columns)

    missing_value_rate = (
        missing_values / total_required_values if total_required_values > 0 else 0.0
    )

    # 2. Duplicate resume rate
    duplicate_count = data.duplicated(subset=["Resume_str"]).sum()

    duplicate_record_rate = duplicate_count / total_rows if total_rows > 0 else 0.0

    # 3. Schema validation rate
    valid_rows = (
        data["Resume_str"].notna()
        & data["Category"].notna()
        & data["Resume_str"].astype(str).str.strip().ne("")
        & data["Category"].astype(str).str.strip().ne("")
    )

    schema_validation_rate = valid_rows.sum() / total_rows if total_rows > 0 else 0.0

    # 4. Class imbalance ratio
    class_counts = data["Category"].value_counts()

    largest_class_size = class_counts.max()
    smallest_class_size = class_counts.min()

    class_imbalance_ratio = (
        largest_class_size / smallest_class_size if smallest_class_size > 0 else 0.0
    )

    metrics = {
        "Missing Value Rate": missing_value_rate,
        "Duplicate Record Rate": duplicate_record_rate,
        "Schema Validation Rate": schema_validation_rate,
        "Class Imbalance Ratio": class_imbalance_ratio,
    }

    results = pd.DataFrame(
        {
            "Metric": metrics.keys(),
            "Value": metrics.values(),
        }
    )

    output_path = PROJECT_ROOT / "artifacts" / "production" / "data_quality_metrics.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\nData Quality Metrics")
    print("-" * 40)
    print(
        f"Missing Value Rate: "
        f"{missing_value_rate:.4f} "
        f"({missing_value_rate * 100:.2f}%)"
    )
    print(
        f"Duplicate Record Rate: "
        f"{duplicate_record_rate:.4f} "
        f"({duplicate_record_rate * 100:.2f}%)"
    )
    print(
        f"Schema Validation Rate: "
        f"{schema_validation_rate:.4f} "
        f"({schema_validation_rate * 100:.2f}%)"
    )
    print(f"Class Imbalance Ratio: " f"{class_imbalance_ratio:.2f}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    calculate_model_quality_metrics()
    calculate_data_quality_metrics()
