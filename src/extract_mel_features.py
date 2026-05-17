import argparse

import librosa
import numpy as np
from tqdm import tqdm

from config import (
    ASVSPOOF_TRAIN_FLAC_DIR,
    ASVSPOOF_DEV_FLAC_DIR,
    ASVSPOOF_EVAL_FLAC_DIR,
    SUBSET_CSV,
    DEV_SUBSET_CSV,
    EVAL_SUBSET_CSV,
    MEL_TRAIN_FEATURES_NPZ,
    MEL_DEV_FEATURES_NPZ,
    MEL_EVAL_FEATURES_NPZ,
    MEL_N_MELS,
    MEL_MAX_FRAMES,
    MEL_N_FFT,
    MEL_HOP_LENGTH,
    SAMPLE_RATE,
    METRICS_DIR,
)

from utils import get_audio_path, load_subset, save_json


def pad_or_truncate_spectrogram(mel_db: np.ndarray, max_frames: int) -> np.ndarray:
    current_frames = mel_db.shape[1]

    if current_frames > max_frames:
        return mel_db[:, :max_frames]

    if current_frames < max_frames:
        pad_width = max_frames - current_frames
        return np.pad(
            mel_db,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=0,
        )

    return mel_db


def extract_mel_spectrogram(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=MEL_N_MELS,
        n_fft=MEL_N_FFT,
        hop_length=MEL_HOP_LENGTH,
        power=2.0,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize from approximately [-80, 0] dB to [0, 1]
    mel_db = np.clip(mel_db, -80.0, 0.0)
    mel_norm = (mel_db + 80.0) / 80.0

    mel_fixed = pad_or_truncate_spectrogram(
        mel_norm,
        max_frames=MEL_MAX_FRAMES,
    )

    # CNN expects channel dimension: [1, n_mels, max_frames]
    mel_fixed = np.expand_dims(mel_fixed, axis=0)

    return mel_fixed.astype(np.float32)


def get_split_config(split_name: str):
    split_configs = {
        "train": {
            "subset_csv": SUBSET_CSV,
            "flac_dir": ASVSPOOF_TRAIN_FLAC_DIR,
            "output_npz": MEL_TRAIN_FEATURES_NPZ,
            "summary_name": "mel_train_feature_extraction_summary.json",
        },
        "dev": {
            "subset_csv": DEV_SUBSET_CSV,
            "flac_dir": ASVSPOOF_DEV_FLAC_DIR,
            "output_npz": MEL_DEV_FEATURES_NPZ,
            "summary_name": "mel_dev_feature_extraction_summary.json",
        },
        "eval": {
            "subset_csv": EVAL_SUBSET_CSV,
            "flac_dir": ASVSPOOF_EVAL_FLAC_DIR,
            "output_npz": MEL_EVAL_FEATURES_NPZ,
            "summary_name": "mel_eval_feature_extraction_summary.json",
        },
    }

    return split_configs[split_name]


def extract_split(split_name: str):
    split_config = get_split_config(split_name)

    subset_csv = split_config["subset_csv"]
    flac_dir = split_config["flac_dir"]
    output_npz = split_config["output_npz"]
    summary_path = METRICS_DIR / split_config["summary_name"]

    print("=" * 60)
    print(f"Extracting Mel Spectrogram Features: {split_name.upper()}")
    print("=" * 60)
    print(f"Subset CSV : {subset_csv}")
    print(f"FLAC dir   : {flac_dir}")
    print(f"Output NPZ : {output_npz}")
    print()

    subset_df = load_subset(subset_csv)

    X = []
    y = []
    audio_files = []
    labels = []
    missing_files = []

    for row in tqdm(subset_df.itertuples(index=False), total=len(subset_df)):
        audio_path = get_audio_path(flac_dir, row.audio_file)

        if not audio_path.exists():
            missing_files.append(str(audio_path))
            continue

        mel_features = extract_mel_spectrogram(audio_path)

        X.append(mel_features)
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

    output_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_npz,
        X=X,
        y=y,
        audio_files=np.array(audio_files),
        labels=np.array(labels),
    )

    summary = {
        "split": split_name,
        "feature_type": "mel_spectrogram",
        "sample_rate": SAMPLE_RATE,
        "n_mels": MEL_N_MELS,
        "max_frames": MEL_MAX_FRAMES,
        "n_fft": MEL_N_FFT,
        "hop_length": MEL_HOP_LENGTH,
        "n_samples": int(X.shape[0]),
        "input_shape": list(X.shape[1:]),
        "n_bonafide": int((y == 0).sum()),
        "n_spoof": int((y == 1).sum()),
        "output_file": str(output_npz),
    }

    save_json(summary, summary_path)

    print()
    print(f"{split_name.upper()} mel feature extraction completed successfully.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Saved to: {output_npz}")
    print()
    print("Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract fixed-size mel spectrogram features for CNN training."
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "eval", "all"],
        required=True,
        help="Which split to process.",
    )

    args = parser.parse_args()

    if args.split == "all":
        for split_name in ["train", "dev", "eval"]:
            extract_split(split_name)
    else:
        extract_split(args.split)


if __name__ == "__main__":
    main()