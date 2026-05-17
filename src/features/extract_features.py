import numpy as np
import librosa
from tqdm import tqdm

from src.config import (
    ASVSPOOF_TRAIN_FLAC_DIR,
    FEATURES_NPZ,
    METRICS_DIR,
    N_MFCC,
    SAMPLE_RATE,
    SUBSET_CSV,
)

from src.utils import get_audio_path, load_subset, save_json


def extract_features(file_path):
    """
    Extract MFCC, delta, and delta-delta features from one audio file.

    Output:
        120-dimensional feature vector:
        - MFCC mean: 20
        - MFCC std: 20
        - Delta mean: 20
        - Delta std: 20
        - Delta-delta mean: 20
        - Delta-delta std: 20
    """

    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {file_path}")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
    )

    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
            np.mean(delta_delta, axis=1),
            np.std(delta_delta, axis=1),
        ]
    )

    return features.astype(np.float32)


def main():
    print("=" * 60)
    print("ASVspoof Feature Extraction")
    print("=" * 60)

    subset_df = load_subset(SUBSET_CSV)

    X = []
    y = []
    audio_files = []
    labels = []

    missing_files = []

    for row in tqdm(subset_df.itertuples(index=False), total=len(subset_df)):
        audio_path = get_audio_path(ASVSPOOF_TRAIN_FLAC_DIR, row.audio_file)

        if not audio_path.exists():
            missing_files.append(str(audio_path))
            continue

        features = extract_features(audio_path)

        X.append(features)
        y.append(int(row.label_id))
        audio_files.append(row.audio_file)
        labels.append(row.label)

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} audio files are missing. "
            f"First missing file: {missing_files[0]}"
        )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    FEATURES_NPZ.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        FEATURES_NPZ,
        X=X,
        y=y,
        audio_files=np.array(audio_files),
        labels=np.array(labels),
    )

    summary = {
        "sample_rate": SAMPLE_RATE,
        "n_mfcc": N_MFCC,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_bonafide": int((y == 0).sum()),
        "n_spoof": int((y == 1).sum()),
        "output_file": str(FEATURES_NPZ),
    }

    save_json(summary, METRICS_DIR / "feature_extraction_summary.json")

    print()
    print("Feature extraction completed successfully.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Saved to: {FEATURES_NPZ}")
    print()
    print("Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()