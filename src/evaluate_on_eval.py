import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    EVAL_FEATURES_NPZ,
    RF_MODEL_PATH,
    MLP_MODEL_PATH,
    MLP_SCALER_PATH,
    METRICS_DIR,
    FIGURES_DIR,
)

from train_mlp import SpoofDetectorMLP


def evaluate_random_forest(X_eval):
    if not RF_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Random Forest model not found: {RF_MODEL_PATH}\n"
            "Run this first: python src\\train_random_forest.py"
        )

    model = joblib.load(RF_MODEL_PATH)
    y_pred = model.predict(X_eval)

    return y_pred


def evaluate_mlp(X_eval):
    if not MLP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MLP model not found: {MLP_MODEL_PATH}\n"
            "Run this first: python src\\train_mlp.py"
        )

    if not MLP_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"MLP scaler not found: {MLP_SCALER_PATH}\n"
            "Run this first: python src\\train_mlp.py"
        )

    scaler = joblib.load(MLP_SCALER_PATH)
    X_eval_scaled = scaler.transform(X_eval)

    input_dim = X_eval_scaled.shape[1]

    model = SpoofDetectorMLP(input_dim=input_dim)
    model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location="cpu"))
    model.eval()

    X_eval_t = torch.FloatTensor(X_eval_scaled)

    with torch.no_grad():
        logits = model(X_eval_t)
        y_pred = logits.argmax(dim=1).numpy()

    return y_pred


def calculate_metrics(model_name, y_true, y_pred):
    return {
        "model": model_name,
        "dataset": "ASVspoof 2019 LA eval balanced subset",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spoof": float(precision_score(y_true, y_pred, average="binary")),
        "recall_spoof": float(recall_score(y_true, y_pred, average="binary")),
        "f1_spoof": float(f1_score(y_true, y_pred, average="binary")),
        "n_samples": int(len(y_true)),
        "n_bonafide": int((y_true == 0).sum()),
        "n_spoof": int((y_true == 1).sum()),
    }


def save_confusion_matrix(y_true, y_pred, title, output_path):
    target_names = ["bonafide", "spoof"]

    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names,
    )

    display.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    print("=" * 60)
    print("Evaluating Models on ASVspoof Eval Subset")
    print("=" * 60)

    if not EVAL_FEATURES_NPZ.exists():
        raise FileNotFoundError(
            f"Eval feature file not found: {EVAL_FEATURES_NPZ}\n"
            "Run this first: python src\\extract_eval_features.py"
        )

    data = np.load(EVAL_FEATURES_NPZ, allow_pickle=True)
    X_eval = data["X"]
    y_eval = data["y"]

    print(f"X_eval shape: {X_eval.shape}")
    print(f"y_eval shape: {y_eval.shape}")
    print()

    target_names = ["bonafide", "spoof"]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Random Forest
    print("-" * 60)
    print("Random Forest Eval Evaluation")
    print("-" * 60)

    rf_pred = evaluate_random_forest(X_eval)

    print(classification_report(y_eval, rf_pred, target_names=target_names))

    rf_metrics = calculate_metrics(
        model_name="Random Forest",
        y_true=y_eval,
        y_pred=rf_pred,
    )

    rf_metrics_path = METRICS_DIR / "random_forest_eval_metrics.json"

    with open(rf_metrics_path, "w", encoding="utf-8") as file:
        json.dump(rf_metrics, file, indent=4)

    save_confusion_matrix(
        y_true=y_eval,
        y_pred=rf_pred,
        title="Random Forest Eval Confusion Matrix",
        output_path=FIGURES_DIR / "random_forest_eval_confusion_matrix.png",
    )

    # MLP
    print("-" * 60)
    print("PyTorch MLP Eval Evaluation")
    print("-" * 60)

    mlp_pred = evaluate_mlp(X_eval)

    print(classification_report(y_eval, mlp_pred, target_names=target_names))

    mlp_metrics = calculate_metrics(
        model_name="PyTorch MLP",
        y_true=y_eval,
        y_pred=mlp_pred,
    )

    mlp_metrics_path = METRICS_DIR / "mlp_eval_metrics.json"

    with open(mlp_metrics_path, "w", encoding="utf-8") as file:
        json.dump(mlp_metrics, file, indent=4)

    save_confusion_matrix(
        y_true=y_eval,
        y_pred=mlp_pred,
        title="PyTorch MLP Eval Confusion Matrix",
        output_path=FIGURES_DIR / "mlp_eval_confusion_matrix.png",
    )

    comparison = [
        rf_metrics,
        mlp_metrics,
    ]

    comparison_path = METRICS_DIR / "eval_model_comparison.json"

    with open(comparison_path, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4)

    print("=" * 60)
    print("Eval Evaluation Summary")
    print("=" * 60)

    for metrics in comparison:
        print(f"Model: {metrics['model']}")
        print(f"Accuracy       : {metrics['accuracy']:.4f}")
        print(f"Spoof Precision: {metrics['precision_spoof']:.4f}")
        print(f"Spoof Recall   : {metrics['recall_spoof']:.4f}")
        print(f"Spoof F1       : {metrics['f1_spoof']:.4f}")
        print()

    print(f"Saved RF metrics to: {rf_metrics_path}")
    print(f"Saved MLP metrics to: {mlp_metrics_path}")
    print(f"Saved eval comparison to: {comparison_path}")


if __name__ == "__main__":
    main()