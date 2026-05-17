import json

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURES_NPZ,
    RF_MODEL_PATH,
    METRICS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
)


def main():
    print("=" * 60)
    print("Training Random Forest Baseline")
    print("=" * 60)

    if not FEATURES_NPZ.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_NPZ}\n"
            "Run this first: python src\\extract_features.py"
        )

    data = np.load(FEATURES_NPZ, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train size: {len(X_train)}")
    print(f"Test size : {len(X_test)}")
    print()

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    target_names = ["bonafide", "spoof"]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    metrics = {
        "model": "RandomForestClassifier",
        "n_estimators": 200,
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "accuracy": float(accuracy),
        "precision_spoof": float(precision),
        "recall_spoof": float(recall),
        "f1_spoof": float(f1),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "n_features": int(X.shape[1]),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    metrics_path = METRICS_DIR / "random_forest_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    joblib.dump(model, RF_MODEL_PATH)

    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names,
    )

    display.plot(values_format="d")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()

    confusion_matrix_path = FIGURES_DIR / "random_forest_confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close()

    print("Training completed successfully.")
    print(f"Model saved to: {RF_MODEL_PATH}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    print()
    print("Summary:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()