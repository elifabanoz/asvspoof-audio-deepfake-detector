import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from config import (
    RF_MODEL_PATH,
    MLP_MODEL_PATH,
    MLP_SCALER_PATH,
)

from extract_features import extract_features
from train_mlp import SpoofDetectorMLP
from utils import ID_TO_LABEL


def predict_with_random_forest(audio_path: Path):
    if not RF_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Random Forest model not found: {RF_MODEL_PATH}\n"
            "Run this first: python src\\train_random_forest.py"
        )

    model = joblib.load(RF_MODEL_PATH)

    features = extract_features(audio_path).reshape(1, -1)

    probabilities = model.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))

    predicted_label_id = int(model.classes_[best_index])
    confidence = float(probabilities[best_index])

    return predicted_label_id, confidence, probabilities


def predict_with_mlp(audio_path: Path):
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

    features = extract_features(audio_path).reshape(1, -1)
    features_scaled = scaler.transform(features)

    input_dim = features_scaled.shape[1]

    model = SpoofDetectorMLP(input_dim=input_dim)
    model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location="cpu"))
    model.eval()

    X_t = torch.FloatTensor(features_scaled)

    with torch.no_grad():
        logits = model(X_t)
        probabilities = torch.softmax(logits, dim=1).numpy()[0]

    predicted_label_id = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_label_id])

    return predicted_label_id, confidence, probabilities


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether an audio file is bonafide or spoof."
    )

    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to a .flac or audio file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["rf", "mlp"],
        default="mlp",
        help="Model to use: rf or mlp. Default: mlp.",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("=" * 60)
    print("Audio Deepfake Prediction")
    print("=" * 60)
    print(f"Audio file: {audio_path}")
    print(f"Model     : {args.model}")
    print()

    if args.model == "rf":
        predicted_label_id, confidence, probabilities = predict_with_random_forest(audio_path)
    else:
        predicted_label_id, confidence, probabilities = predict_with_mlp(audio_path)

    predicted_label = ID_TO_LABEL[predicted_label_id]

    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    print()
    print("Class probabilities:")
    print(f"bonafide: {probabilities[0]:.4f}")
    print(f"spoof   : {probabilities[1]:.4f}")


if __name__ == "__main__":
    main()