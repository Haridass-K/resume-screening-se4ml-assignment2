import random
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "production" / "resume_screening_model.pkl"
)

CANDIDATE_MODEL_PATH = PROJECT_ROOT / "models" / "research" / "logistic_regression.pkl"

OUTPUT_PATH = (
    PROJECT_ROOT / "artifacts" / "production" / "shadow_deployment_results.csv"
)


def get_prediction(model, resume_text):
    """Return prediction, confidence, and inference latency."""

    start_time = perf_counter()

    predicted_category = model.predict([resume_text])[0]

    probabilities = model.predict_proba([resume_text])[0]
    confidence = probabilities.max() * 100

    latency_ms = (perf_counter() - start_time) * 1000

    return {
        "predicted_category": predicted_category,
        "confidence": round(confidence, 2),
        "latency_ms": round(latency_ms, 2),
    }


def run_shadow_deployment(resume_text):
    """
    Run production and candidate models on the same input.

    Only the production prediction is treated as the user-facing output.
    The candidate prediction is recorded for comparison.
    """

    production_model = joblib.load(PRODUCTION_MODEL_PATH)
    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)

    production_result = get_prediction(
        production_model,
        resume_text,
    )

    candidate_result = get_prediction(
        candidate_model,
        resume_text,
    )

    agreement = (
        production_result["predicted_category"]
        == candidate_result["predicted_category"]
    )

    result = {
        "production_prediction": production_result["predicted_category"],
        "production_confidence": production_result["confidence"],
        "production_latency_ms": production_result["latency_ms"],
        "candidate_prediction": candidate_result["predicted_category"],
        "candidate_confidence": candidate_result["confidence"],
        "candidate_latency_ms": candidate_result["latency_ms"],
        "prediction_agreement": agreement,
    }

    results_df = pd.DataFrame([result])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nShadow Deployment Results")
    print("-" * 50)

    print(
        "User-facing production prediction:",
        production_result["predicted_category"],
    )

    print(
        "Production confidence:",
        f"{production_result['confidence']:.2f}%",
    )

    print(
        "Production latency:",
        f"{production_result['latency_ms']:.2f} ms",
    )

    print(
        "\nHidden candidate prediction:",
        candidate_result["predicted_category"],
    )

    print(
        "Candidate confidence:",
        f"{candidate_result['confidence']:.2f}%",
    )

    print(
        "Candidate latency:",
        f"{candidate_result['latency_ms']:.2f} ms",
    )

    print("\nPrediction agreement:", agreement)
    print(f"\nResults saved to: {OUTPUT_PATH}")


def run_canary_release(resume_text, candidate_traffic_percentage=10):
    """
    Route a small percentage of requests to the candidate model.

    The remaining traffic continues to use the production model.
    """

    production_model = joblib.load(PRODUCTION_MODEL_PATH)
    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)

    random_number = random.randint(1, 100)

    if random_number <= candidate_traffic_percentage:
        selected_model = "candidate"
        result = get_prediction(candidate_model, resume_text)
    else:
        selected_model = "production"
        result = get_prediction(production_model, resume_text)

    canary_result = {
        "selected_model": selected_model,
        "candidate_traffic_percentage": candidate_traffic_percentage,
        "predicted_category": result["predicted_category"],
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
    }

    output_path = (
        PROJECT_ROOT / "artifacts" / "production" / "canary_release_results.csv"
    )

    pd.DataFrame([canary_result]).to_csv(
        output_path,
        index=False,
    )

    print("\nCanary Release Results")
    print("-" * 50)
    print("Candidate traffic percentage:", f"{candidate_traffic_percentage}%")
    print("Selected model:", selected_model)
    print("Prediction:", result["predicted_category"])
    print("Confidence:", f"{result['confidence']:.2f}%")
    print("Latency:", f"{result['latency_ms']:.2f} ms")
    print(f"\nResults saved to: {output_path}")


def run_ab_testing(resume_text):
    """
    Compare Production Model (A) and Candidate Model (B)
    using the same resume input.
    """

    production_model = joblib.load(PRODUCTION_MODEL_PATH)
    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)

    production_result = get_prediction(
        production_model,
        resume_text,
    )

    candidate_result = get_prediction(
        candidate_model,
        resume_text,
    )

    ab_result = {
        "group_a_model": "Production",
        "group_a_prediction": production_result["predicted_category"],
        "group_a_confidence": production_result["confidence"],
        "group_a_latency_ms": production_result["latency_ms"],
        "group_b_model": "Candidate",
        "group_b_prediction": candidate_result["predicted_category"],
        "group_b_confidence": candidate_result["confidence"],
        "group_b_latency_ms": candidate_result["latency_ms"],
        "prediction_agreement": (
            production_result["predicted_category"]
            == candidate_result["predicted_category"]
        ),
    }

    output_path = PROJECT_ROOT / "artifacts" / "production" / "ab_testing_results.csv"

    pd.DataFrame([ab_result]).to_csv(
        output_path,
        index=False,
    )

    print("\nA/B Testing Results")
    print("-" * 50)

    print("Group A (Production)")
    print(f"Prediction : {production_result['predicted_category']}")
    print(f"Confidence: {production_result['confidence']:.2f}%")
    print(f"Latency   : {production_result['latency_ms']:.2f} ms")

    print("\nGroup B (Candidate)")
    print(f"Prediction : {candidate_result['predicted_category']}")
    print(f"Confidence: {candidate_result['confidence']:.2f}%")
    print(f"Latency   : {candidate_result['latency_ms']:.2f} ms")

    print(
        "\nPrediction Agreement:",
        ab_result["prediction_agreement"],
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    sample_resume = (
        "Experienced Python developer with machine learning, SQL, "
        "data analysis, APIs, cloud deployment, and software engineering."
    )

    print("=" * 70)
    print("1. Shadow Deployment")
    print("=" * 70)
    run_shadow_deployment(sample_resume)

    print("\n" + "=" * 70)
    print("2. Canary Release")
    print("=" * 70)
    run_canary_release(
        sample_resume,
        candidate_traffic_percentage=10,
    )

    print("\n" + "=" * 70)
    print("3. A/B Testing")
    print("=" * 70)
    run_ab_testing(sample_resume)
