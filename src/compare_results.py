import json

import pandas as pd

from config import METRICS_DIR


def load_metrics(file_name: str) -> dict:
    metrics_path = METRICS_DIR / file_name

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}\n"
            "Run the training scripts first."
        )

    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)

    rf_metrics = load_metrics("random_forest_metrics.json")
    mlp_metrics = load_metrics("mlp_metrics.json")

    comparison = [
        {
            "model": "Random Forest",
            "accuracy": rf_metrics["accuracy"],
            "spoof_precision": rf_metrics["precision_spoof"],
            "spoof_recall": rf_metrics["recall_spoof"],
            "spoof_f1": rf_metrics["f1_spoof"],
        },
        {
            "model": "PyTorch MLP",
            "accuracy": mlp_metrics["accuracy"],
            "spoof_precision": mlp_metrics["precision_spoof"],
            "spoof_recall": mlp_metrics["recall_spoof"],
            "spoof_f1": mlp_metrics["f1_spoof"],
        },
    ]

    comparison_df = pd.DataFrame(comparison)

    print()
    print(comparison_df.to_string(index=False))
    print()

    csv_path = METRICS_DIR / "model_comparison.csv"
    json_path = METRICS_DIR / "model_comparison.json"

    comparison_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4)

    best_model = comparison_df.sort_values(
        by="spoof_f1",
        ascending=False,
    ).iloc[0]

    print(f"Best model by spoof F1: {best_model['model']}")
    print(f"Spoof F1: {best_model['spoof_f1']:.4f}")
    print()
    print(f"Comparison CSV saved to: {csv_path}")
    print(f"Comparison JSON saved to: {json_path}")


if __name__ == "__main__":
    main()