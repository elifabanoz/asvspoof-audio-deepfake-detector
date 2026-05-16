import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import (
    FEATURES_NPZ,
    FIGURES_DIR,
    METRICS_DIR,
    MLP_MODEL_PATH,
    MLP_SCALER_PATH,
    RANDOM_STATE,
)


class SpoofDetectorMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("=" * 60)
    print("Training PyTorch MLP Baseline")
    print("=" * 60)

    if not FEATURES_NPZ.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_NPZ}\n"
            "Run this first: python src\\extract_features.py"
        )

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)

    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=64,
        shuffle=True,
    )

    input_dim = X_train.shape[1]
    model = SpoofDetectorMLP(input_dim=input_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 30

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            y_pred_t = test_outputs.argmax(dim=1)
            test_acc = (y_pred_t == y_test_t).float().mean().item()

        avg_loss = total_loss / len(train_loader)

        train_losses.append(avg_loss)
        test_accuracies.append(test_acc)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"loss={avg_loss:.4f} | "
            f"test_acc={test_acc:.4f}"
        )

    model.eval()
    with torch.no_grad():
        final_outputs = model(X_test_t)
        y_pred = final_outputs.argmax(dim=1).numpy()

    target_names = ["bonafide", "spoof"]

    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")

    metrics = {
        "model": "PyTorch MLP",
        "epochs": epochs,
        "batch_size": 64,
        "learning_rate": 1e-3,
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
    MLP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), MLP_MODEL_PATH)
    joblib.dump(scaler, MLP_SCALER_PATH)

    metrics_path = METRICS_DIR / "mlp_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names,
    )

    display.plot(values_format="d")
    plt.title("PyTorch MLP Confusion Matrix")
    plt.tight_layout()

    confusion_matrix_path = FIGURES_DIR / "mlp_confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("MLP Training Loss")
    plt.tight_layout()

    loss_curve_path = FIGURES_DIR / "mlp_training_loss.png"
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(range(1, epochs + 1), test_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("MLP Test Accuracy")
    plt.tight_layout()

    accuracy_curve_path = FIGURES_DIR / "mlp_test_accuracy.png"
    plt.savefig(accuracy_curve_path, dpi=300)
    plt.close()

    print("Training completed successfully.")
    print(f"Model saved to: {MLP_MODEL_PATH}")
    print(f"Scaler saved to: {MLP_SCALER_PATH}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    print(f"Loss curve saved to: {loss_curve_path}")
    print(f"Accuracy curve saved to: {accuracy_curve_path}")
    print()

    print("Summary:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()