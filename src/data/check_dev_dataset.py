import pandas as pd

from src.config import (
    ASVSPOOF_DEV_DIR,
    ASVSPOOF_DEV_FLAC_DIR,
    ASVSPOOF_DEV_LABELS,
)


def main():
    print("=" * 60)
    print("ASVspoof Dev Dataset Check")
    print("=" * 60)

    print(f"Dev directory : {ASVSPOOF_DEV_DIR}")
    print(f"FLAC directory: {ASVSPOOF_DEV_FLAC_DIR}")
    print(f"Label file    : {ASVSPOOF_DEV_LABELS}")
    print()

    dev_exists = ASVSPOOF_DEV_DIR.exists()
    flac_exists = ASVSPOOF_DEV_FLAC_DIR.exists()
    labels_exists = ASVSPOOF_DEV_LABELS.exists()

    print(f"Dev directory found : {dev_exists}")
    print(f"FLAC directory found: {flac_exists}")
    print(f"Label file found    : {labels_exists}")
    print()

    if not dev_exists or not flac_exists or not labels_exists:
        print("Dev dataset path is not correct. Please check data/raw folder structure.")
        return

    audio_files = list(ASVSPOOF_DEV_FLAC_DIR.glob("*.flac"))

    labels_df = pd.read_csv(
        ASVSPOOF_DEV_LABELS,
        sep=r"\s+",
        header=None,
        names=["speaker_id", "audio_file", "unused", "system_id", "label"],
    )

    print(f"Number of audio files: {len(audio_files)}")
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
        print("Dev dataset check passed successfully.")
    else:
        print("Some files listed in the dev protocol are missing from the flac folder.")
        print("Example missing files:")
        print(list(missing_audio_files)[:10])


if __name__ == "__main__":
    main()