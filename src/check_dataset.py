from pathlib import Path

import pandas as pd

from config import (
    ASVSPOOF_TRAIN_DIR,
    ASVSPOOF_TRAIN_FLAC_DIR,
    ASVSPOOF_TRAIN_LABELS,
)


def main():
    print("=" * 60)
    print("ASVspoof Dataset Check")
    print("=" * 60)

    print(f"Train directory: {ASVSPOOF_TRAIN_DIR}")
    print(f"FLAC directory : {ASVSPOOF_TRAIN_FLAC_DIR}")
    print(f"Label file     : {ASVSPOOF_TRAIN_LABELS}")
    print()

    train_exists = ASVSPOOF_TRAIN_DIR.exists()
    flac_exists = ASVSPOOF_TRAIN_FLAC_DIR.exists()
    labels_exists = ASVSPOOF_TRAIN_LABELS.exists()

    print(f"Train directory found: {train_exists}")
    print(f"FLAC directory found : {flac_exists}")
    print(f"Label file found     : {labels_exists}")
    print()

    if not train_exists or not flac_exists or not labels_exists:
        print("Dataset path is not correct. Please check data/raw folder structure.")
        return

    audio_files = list(ASVSPOOF_TRAIN_FLAC_DIR.glob("*.flac"))
    print(f"Number of audio files: {len(audio_files)}")

    labels_df = pd.read_csv(
        ASVSPOOF_TRAIN_LABELS,
        sep=r"\s+",
        header=None,
        names=["speaker_id", "audio_file", "unused", "system_id", "label"],
    )

    print(f"Number of label rows : {len(labels_df)}")
    print()

    print("Label distribution:")
    print(labels_df["label"].value_counts())
    print()

    print("First 5 rows:")
    print(labels_df.head())
    print()

    audio_file_stems = {path.stem for path in audio_files}
    label_file_names = set(labels_df["audio_file"])

    missing_audio_files = label_file_names - audio_file_stems

    print(f"Missing audio files according to labels: {len(missing_audio_files)}")

    if len(missing_audio_files) == 0:
        print("Dataset check passed successfully.")
    else:
        print("Some files listed in the protocol are missing from the flac folder.")
        print("Example missing files:")
        print(list(missing_audio_files)[:10])


if __name__ == "__main__":
    main()