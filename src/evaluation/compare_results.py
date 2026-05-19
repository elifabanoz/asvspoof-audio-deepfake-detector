import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import METRICS_DIR, FIGURES_DIR


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Metric file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_comparison_dataframe():
    rf_train = load_json(METRICS_DIR / "random_forest_metrics.json")
    mlp_train = load_json(METRICS_DIR / "mlp_metrics.json")
    cnn_train = load_json(METRICS_DIR / "cnn_metrics.json")

    rf_dev = load_json(METRICS_DIR / "random_forest_dev_metrics.json")
    mlp_dev = load_json(METRICS_DIR / "mlp_dev_metrics.json")
    cnn_dev = load_json(METRICS_DIR / "cnn_dev_metrics.json")

    rf_eval = load_json(METRICS_DIR / "random_forest_eval_metrics.json")
    mlp_eval = load_json(METRICS_DIR / "mlp_eval_metrics.json")
    cnn_eval = load_json(METRICS_DIR / "cnn_eval_metrics.json")

    rows = [
        {
            "model": "Random Forest",
            "feature_type": "MFCC statistics",
            "train_accuracy": rf_train["accuracy"],
            "dev_accuracy": rf_dev["accuracy"],
            "eval_accuracy": rf_eval["accuracy"],
            "eval_spoof_precision": rf_eval["precision_spoof"],
            "eval_spoof_recall": rf_eval["recall_spoof"],
            "eval_spoof_f1": rf_eval["f1_spoof"],
        },
        {
            "model": "PyTorch MLP",
            "feature_type": "MFCC statistics",
            "train_accuracy": mlp_train["accuracy"],
            "dev_accuracy": mlp_dev["accuracy"],
            "eval_accuracy": mlp_eval["accuracy"],
            "eval_spoof_precision": mlp_eval["precision_spoof"],
            "eval_spoof_recall": mlp_eval["recall_spoof"],
            "eval_spoof_f1": mlp_eval["f1_spoof"],
        },
        {
            "model": "Spectrogram CNN",
            "feature_type": "Mel spectrogram",
            "train_accuracy": cnn_train["accuracy"],
            "dev_accuracy": cnn_dev["accuracy"],
            "eval_accuracy": cnn_eval["accuracy"],
            "eval_spoof_precision": cnn_eval["precision_spoof"],
            "eval_spoof_recall": cnn_eval["recall_spoof"],
            "eval_spoof_f1": cnn_eval["f1_spoof"],
        },
    ]

    return pd.DataFrame(rows)


def save_accuracy_plot(df: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = df.set_index("model")[
        ["train_accuracy", "dev_accuracy", "eval_accuracy"]
    ]

    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Model Accuracy Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(["Train Split", "Dev Subset", "Eval Subset"])
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = FIGURES_DIR / "model_accuracy_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_eval_spoof_metrics_plot(df: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = df.set_index("model")[
        ["eval_spoof_precision", "eval_spoof_recall", "eval_spoof_f1"]
    ]

    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Eval Spoof Metrics Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(["Precision", "Recall", "F1"])
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = FIGURES_DIR / "eval_spoof_metrics_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("Model Results Comparison")
    print("=" * 60)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_comparison_dataframe()

    csv_path = METRICS_DIR / "model_comparison.csv"
    json_path = METRICS_DIR / "model_comparison.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=4)

    accuracy_plot_path = save_accuracy_plot(df)
    spoof_plot_path = save_eval_spoof_metrics_plot(df)

    print(df.to_string(index=False))
    print()
    print(f"Saved CSV comparison to: {csv_path}")
    print(f"Saved JSON comparison to: {json_path}")
    print(f"Saved accuracy plot to: {accuracy_plot_path}")
    print(f"Saved eval spoof metrics plot to: {spoof_plot_path}")


if __name__ == "__main__":
    main()