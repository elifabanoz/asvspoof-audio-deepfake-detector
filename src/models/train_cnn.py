import json
import random
import copy

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
from torch.utils.data import DataLoader, Dataset

from src.config import (
    CNN_MODEL_PATH,
    FIGURES_DIR,
    MEL_TRAIN_FEATURES_NPZ,
    METRICS_DIR,
    RANDOM_STATE,
)


class MelSpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        predictions = logits.argmax(dim=1)
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            predictions = logits.argmax(dim=1)

            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_labels), np.array(all_predictions)


def save_training_curves(train_losses, val_losses, train_accs, val_accs):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    loss_path = FIGURES_DIR / "cnn_training_loss.png"
    acc_path = FIGURES_DIR / "cnn_validation_accuracy.png"

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()

    return loss_path, acc_path


def save_confusion_matrix(y_true, y_pred):
    output_path = FIGURES_DIR / "cnn_confusion_matrix.png"

    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["bonafide", "spoof"],
    )

    display.plot(values_format="d")
    plt.title("CNN Train Split Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("Training CNN on Mel Spectrogram Features")
    print("=" * 60)

    set_seed(RANDOM_STATE)

    if not MEL_TRAIN_FEATURES_NPZ.exists():
        raise FileNotFoundError(
            f"Mel train feature file not found: {MEL_TRAIN_FEATURES_NPZ}\n"
            "Run this first: python src\\extract_mel_features.py --split train"
        )

    data = np.load(MEL_TRAIN_FEATURES_NPZ, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train size: {len(y_train)}")
    print(f"Validation size: {len(y_val)}")
    print()

    train_dataset = MelSpectrogramDataset(X_train, y_train)
    val_dataset = MelSpectrogramDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    model = SpectrogramCNN(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    epochs = 20

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    best_state_dict = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc, y_true, y_pred = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_val_loss, final_val_acc, y_true, y_pred = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["bonafide", "spoof"],
        )
    )

    CNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CNN_MODEL_PATH)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": "Spectrogram CNN",
        "feature_type": "mel_spectrogram",
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_acc),
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spoof": float(precision_score(y_true, y_pred)),
        "recall_spoof": float(recall_score(y_true, y_pred)),
        "f1_spoof": float(f1_score(y_true, y_pred)),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "input_shape": list(X.shape[1:]),
    }

    metrics_path = METRICS_DIR / "cnn_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    loss_path, acc_path = save_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
    )

    confusion_matrix_path = save_confusion_matrix(y_true, y_pred)

    print("Training completed successfully.")
    print(f"Model saved to: {CNN_MODEL_PATH}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Loss curve saved to: {loss_path}")
    print(f"Accuracy curve saved to: {acc_path}")
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    print()
    print("Summary:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()