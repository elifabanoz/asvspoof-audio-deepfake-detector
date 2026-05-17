import json

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
    CNN_MODEL_PATH,
    MEL_DEV_FEATURES_NPZ,
    MEL_EVAL_FEATURES_NPZ,
    METRICS_DIR,
    FIGURES_DIR,
)

from train_cnn import SpectrogramCNN


def load_cnn_model(device):
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN model not found: {CNN_MODEL_PATH}\n"
            "Run this first: python src\\train_cnn.py"
        )

    model = SpectrogramCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    model.eval()

    return model


def predict(model, X, device, batch_size=32):
    predictions = []

    with torch.no_grad():
        for start_idx in range(0, len(X), batch_size):
            end_idx = start_idx + batch_size

            X_batch = torch.FloatTensor(X[start_idx:end_idx]).to(device)

            logits = model(X_batch)
            batch_predictions = logits.argmax(dim=1).cpu().numpy()

            predictions.extend(batch_predictions)

    return np.array(predictions)


def calculate_metrics(model_name, split_name, y_true, y_pred):
    return {
        "model": model_name,
        "split": split_name,
        "dataset": f"ASVspoof 2019 LA {split_name} balanced subset",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spoof": float(precision_score(y_true, y_pred)),
        "recall_spoof": float(recall_score(y_true, y_pred)),
        "f1_spoof": float(f1_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
        "n_bonafide": int((y_true == 0).sum()),
        "n_spoof": int((y_true == 1).sum()),
    }


def save_confusion_matrix(y_true, y_pred, title, output_path):
    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["bonafide", "spoof"],
    )

    display.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate_split(model, split_name, feature_path, device):
    if not feature_path.exists():
        raise FileNotFoundError(
            f"{split_name} mel feature file not found: {feature_path}\n"
            f"Run this first: python src\\extract_mel_features.py --split {split_name}"
        )

    data = np.load(feature_path, allow_pickle=True)

    X = data["X"]
    y = data["y"]

    print("-" * 60)
    print(f"Spectrogram CNN {split_name.upper()} Evaluation")
    print("-" * 60)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print()

    y_pred = predict(model, X, device)

    print(
        classification_report(
            y,
            y_pred,
            target_names=["bonafide", "spoof"],
        )
    )

    metrics = calculate_metrics(
        model_name="Spectrogram CNN",
        split_name=split_name,
        y_true=y,
        y_pred=y_pred,
    )

    metrics_path = METRICS_DIR / f"cnn_{split_name}_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    confusion_matrix_path = FIGURES_DIR / f"cnn_{split_name}_confusion_matrix.png"

    save_confusion_matrix(
        y_true=y,
        y_pred=y_pred,
        title=f"Spectrogram CNN {split_name.upper()} Confusion Matrix",
        output_path=confusion_matrix_path,
    )

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {confusion_matrix_path}")
    print()

    return metrics


def main():
    print("=" * 60)
    print("Evaluating CNN on ASVspoof Dev and Eval Subsets")
    print("=" * 60)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    model = load_cnn_model(device)

    dev_metrics = evaluate_split(
        model=model,
        split_name="dev",
        feature_path=MEL_DEV_FEATURES_NPZ,
        device=device,
    )

    eval_metrics = evaluate_split(
        model=model,
        split_name="eval",
        feature_path=MEL_EVAL_FEATURES_NPZ,
        device=device,
    )

    comparison = [
        dev_metrics,
        eval_metrics,
    ]

    comparison_path = METRICS_DIR / "cnn_dev_eval_comparison.json"

    with open(comparison_path, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4)

    print("=" * 60)
    print("CNN Dev/Eval Evaluation Summary")
    print("=" * 60)

    for metrics in comparison:
        print(f"Split: {metrics['split']}")
        print(f"Accuracy       : {metrics['accuracy']:.4f}")
        print(f"Spoof Precision: {metrics['precision_spoof']:.4f}")
        print(f"Spoof Recall   : {metrics['recall_spoof']:.4f}")
        print(f"Spoof F1       : {metrics['f1_spoof']:.4f}")
        print()

    print(f"Saved CNN dev/eval comparison to: {comparison_path}")


if __name__ == "__main__":
    main()