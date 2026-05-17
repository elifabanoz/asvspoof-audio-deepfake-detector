import numpy as np
from tqdm import tqdm

from src.config import (
    ASVSPOOF_EVAL_FLAC_DIR,
    EVAL_FEATURES_NPZ,
    EVAL_SUBSET_CSV,
    METRICS_DIR,
    N_MFCC,
    SAMPLE_RATE,
)

from src.features.extract_features import extract_features
from src.utils import get_audio_path, load_subset, save_json


def main():
    print("=" * 60)
    print("ASVspoof Eval Feature Extraction")
    print("=" * 60)

    subset_df = load_subset(EVAL_SUBSET_CSV)

    X = []
    y = []
    audio_files = []
    labels = []

    missing_files = []

    for row in tqdm(subset_df.itertuples(index=False), total=len(subset_df)):
        audio_path = get_audio_path(ASVSPOOF_EVAL_FLAC_DIR, row.audio_file)

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

    EVAL_FEATURES_NPZ.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        EVAL_FEATURES_NPZ,
        X=X,
        y=y,
        audio_files=np.array(audio_files),
        labels=np.array(labels),
    )

    summary = {
        "dataset": "ASVspoof 2019 LA eval subset",
        "sample_rate": SAMPLE_RATE,
        "n_mfcc": N_MFCC,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_bonafide": int((y == 0).sum()),
        "n_spoof": int((y == 1).sum()),
        "output_file": str(EVAL_FEATURES_NPZ),
    }

    save_json(summary, METRICS_DIR / "eval_feature_extraction_summary.json")

    print()
    print("Eval feature extraction completed successfully.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Saved to: {EVAL_FEATURES_NPZ}")
    print()
    print("Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()