import copy
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import (
    FIGURES_DIR,
    MEL_TRAIN_FEATURES_NPZ,
    METRICS_DIR,
    PGD_ADV_CNN_MODEL_PATH,
    RANDOM_STATE,
)

from src.models.train_cnn import SpectrogramCNN


class MelSpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_random_start_pgd_examples(
    model,
    X_batch,
    y_batch,
    epsilon,
    alpha,
    num_steps,
    criterion,
):
    original_X = X_batch.clone().detach()

    random_delta = torch.empty_like(original_X).uniform_(-epsilon, epsilon)
    X_adv = torch.clamp(original_X + random_delta, 0.0, 1.0).detach()

    for _ in range(num_steps):
        X_adv.requires_grad_(True)

        logits = model(X_adv)
        loss = criterion(logits, y_batch)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()

            delta = torch.clamp(
                X_adv - original_X,
                min=-epsilon,
                max=epsilon,
            )

            X_adv = torch.clamp(original_X + delta, 0.0, 1.0).detach()

    return X_adv


def train_one_epoch_pgd_adversarial(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epsilon,
    alpha,
    num_steps,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        X_adv = create_random_start_pgd_examples(
            model=model,
            X_batch=X_batch,
            y_batch=y_batch,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            criterion=criterion,
        )

        optimizer.zero_grad()

        clean_logits = model(X_batch)
        adv_logits = model(X_adv)

        clean_loss = criterion(clean_logits, y_batch)
        adv_loss = criterion(adv_logits, y_batch)

        loss = 0.5 * clean_loss + 0.5 * adv_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        predictions = clean_logits.argmax(dim=1)
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate_clean(model, dataloader, criterion, device):
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

    loss_path = FIGURES_DIR / "pgd_adv_cnn_training_loss.png"
    acc_path = FIGURES_DIR / "pgd_adv_cnn_validation_accuracy.png"

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PGD Adversarial CNN Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("PGD Adversarial CNN Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()

    return loss_path, acc_path


def main():
    print("=" * 60)
    print("Training PGD Adversarially Robust CNN")
    print("=" * 60)

    set_seed(RANDOM_STATE)

    if not MEL_TRAIN_FEATURES_NPZ.exists():
        raise FileNotFoundError(
            f"Mel train feature file not found: {MEL_TRAIN_FEATURES_NPZ}\n"
            "Run this first: python -m src.features.extract_mel_features --split train"
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

    train_dataset = MelSpectrogramDataset(X_train, y_train)
    val_dataset = MelSpectrogramDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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

    epochs = 12
    pgd_epsilon = 0.005
    pgd_alpha = 0.00125
    pgd_steps = 5

    print("PGD adversarial training config:")
    print(f"epsilon: {pgd_epsilon}")
    print(f"alpha  : {pgd_alpha}")
    print(f"steps  : {pgd_steps}")
    print()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    best_epoch = 0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch_pgd_adversarial(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epsilon=pgd_epsilon,
            alpha=pgd_alpha,
            num_steps=pgd_steps,
        )

        val_loss, val_acc, y_true, y_pred = evaluate_clean(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
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

    final_val_loss, final_val_acc, y_true, y_pred = evaluate_clean(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
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

    PGD_ADV_CNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), PGD_ADV_CNN_MODEL_PATH)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": "PGD Adversarially Trained Spectrogram CNN",
        "feature_type": "mel_spectrogram",
        "defense": "random-start PGD adversarial training",
        "pgd_training_epsilon": pgd_epsilon,
        "pgd_training_alpha": pgd_alpha,
        "pgd_training_steps": pgd_steps,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_acc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spoof": float(precision_score(y_true, y_pred)),
        "recall_spoof": float(recall_score(y_true, y_pred)),
        "f1_spoof": float(f1_score(y_true, y_pred)),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "input_shape": list(X.shape[1:]),
    }

    metrics_path = METRICS_DIR / "pgd_adv_cnn_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    loss_path, acc_path = save_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
    )

    print()
    print("PGD adversarial training completed successfully.")
    print(f"Model saved to: {PGD_ADV_CNN_MODEL_PATH}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Loss curve saved to: {loss_path}")
    print(f"Accuracy curve saved to: {acc_path}")
    print()
    print("Summary:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()