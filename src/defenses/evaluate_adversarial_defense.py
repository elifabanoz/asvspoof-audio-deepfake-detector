import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import (
    CNN_MODEL_PATH,
    ADV_CNN_MODEL_PATH,
    MULTI_EPS_ADV_CNN_MODEL_PATH,
    FIGURES_DIR,
    MEL_EVAL_FEATURES_NPZ,
    METRICS_DIR,
)

from src.models.train_cnn import SpectrogramCNN


def load_model(model_path, device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = SpectrogramCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def fgsm_attack(model, X_batch, y_batch, epsilon, criterion):
    X_adv = X_batch.clone().detach().requires_grad_(True)

    logits = model(X_adv)
    loss = criterion(logits, y_batch)

    model.zero_grad()
    loss.backward()

    grad_sign = X_adv.grad.data.sign()
    X_adv = X_adv + epsilon * grad_sign
    X_adv = torch.clamp(X_adv, 0.0, 1.0)

    return X_adv.detach()


def evaluate_model_under_fgsm(model, X, y, epsilon, device, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    predictions = []

    for start_idx in range(0, len(X), batch_size):
        end_idx = start_idx + batch_size

        X_batch = torch.FloatTensor(X[start_idx:end_idx]).to(device)
        y_batch = torch.LongTensor(y[start_idx:end_idx]).to(device)

        if epsilon > 0:
            X_input = fgsm_attack(
                model=model,
                X_batch=X_batch,
                y_batch=y_batch,
                epsilon=epsilon,
                criterion=criterion,
            )
        else:
            X_input = X_batch

        with torch.no_grad():
            logits = model(X_input)
            batch_predictions = logits.argmax(dim=1).cpu().numpy()

        predictions.extend(batch_predictions)

    y_pred = np.array(predictions)

    return {
        "epsilon": float(epsilon),
        "accuracy": float(accuracy_score(y, y_pred)),
        "spoof_precision": float(precision_score(y, y_pred, zero_division=0)),
        "spoof_recall": float(recall_score(y, y_pred, zero_division=0)),
        "spoof_f1": float(f1_score(y, y_pred, zero_division=0)),
    }


def save_accuracy_comparison_plot(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / "fgsm_three_model_accuracy_comparison.png"

    plt.figure(figsize=(9, 5))

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        plt.plot(
            model_df["epsilon"],
            model_df["accuracy"],
            marker="o",
            label=model_name,
        )

    plt.xlabel("FGSM epsilon")
    plt.ylabel("Accuracy")
    plt.title("CNN Defense Comparison Under FGSM")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_spoof_recall_comparison_plot(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / "fgsm_three_model_spoof_recall_comparison.png"

    plt.figure(figsize=(9, 5))

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        plt.plot(
            model_df["epsilon"],
            model_df["spoof_recall"],
            marker="o",
            label=model_name,
        )

    plt.xlabel("FGSM epsilon")
    plt.ylabel("Spoof Recall")
    plt.title("Spoof Recall Defense Comparison Under FGSM")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("Evaluating CNN Adversarial Defense Strategies")
    print("=" * 60)

    if not MEL_EVAL_FEATURES_NPZ.exists():
        raise FileNotFoundError(
            f"Eval mel feature file not found: {MEL_EVAL_FEATURES_NPZ}\n"
            "Run this first: python -m src.features.extract_mel_features --split eval"
        )

    data = np.load(MEL_EVAL_FEATURES_NPZ, allow_pickle=True)
    X_eval = data["X"]
    y_eval = data["y"]

    print(f"X_eval shape: {X_eval.shape}")
    print(f"y_eval shape: {y_eval.shape}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    clean_model = load_model(CNN_MODEL_PATH, device)
    single_eps_adv_model = load_model(ADV_CNN_MODEL_PATH, device)
    multi_eps_adv_model = load_model(MULTI_EPS_ADV_CNN_MODEL_PATH, device)

    epsilons = [0.0, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]

    model_configs = [
        ("Clean CNN", clean_model),
        ("Single-Epsilon Adv CNN", single_eps_adv_model),
        ("Multi-Epsilon Adv CNN", multi_eps_adv_model),
    ]

    all_results = []

    for model_name, model in model_configs:
        print("-" * 60)
        print(model_name)
        print("-" * 60)

        for epsilon in epsilons:
            metrics = evaluate_model_under_fgsm(
                model=model,
                X=X_eval,
                y=y_eval,
                epsilon=epsilon,
                device=device,
            )

            metrics["model"] = model_name
            all_results.append(metrics)

            print(
                f"epsilon={epsilon:.3f} | "
                f"accuracy={metrics['accuracy']:.4f} | "
                f"spoof_precision={metrics['spoof_precision']:.4f} | "
                f"spoof_recall={metrics['spoof_recall']:.4f} | "
                f"spoof_f1={metrics['spoof_f1']:.4f}"
            )

        print()

    df = pd.DataFrame(all_results)

    df = df[
        [
            "model",
            "epsilon",
            "accuracy",
            "spoof_precision",
            "spoof_recall",
            "spoof_f1",
        ]
    ]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = METRICS_DIR / "fgsm_three_model_defense_comparison.csv"
    json_path = METRICS_DIR / "fgsm_three_model_defense_comparison.json"

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4)

    accuracy_plot_path = save_accuracy_comparison_plot(df)
    spoof_recall_plot_path = save_spoof_recall_comparison_plot(df)

    print("=" * 60)
    print("Three-Model Defense Evaluation Summary")
    print("=" * 60)

    print(df.to_string(index=False))
    print()
    print(f"Saved CSV results to: {csv_path}")
    print(f"Saved JSON results to: {json_path}")
    print(f"Saved accuracy comparison plot to: {accuracy_plot_path}")
    print(f"Saved spoof recall comparison plot to: {spoof_recall_plot_path}")


if __name__ == "__main__":
    main()