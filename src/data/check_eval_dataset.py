import pandas as pd

from src.config import (
    ASVSPOOF_EVAL_DIR,
    ASVSPOOF_EVAL_FLAC_DIR,
    ASVSPOOF_EVAL_LABELS,
)


def main():
    print("=" * 60)
    print("ASVspoof Eval Dataset Check")
    print("=" * 60)

    print(f"Eval directory: {ASVSPOOF_EVAL_DIR}")
    print(f"FLAC directory: {ASVSPOOF_EVAL_FLAC_DIR}")
    print(f"Label file    : {ASVSPOOF_EVAL_LABELS}")
    print()

    eval_exists = ASVSPOOF_EVAL_DIR.exists()
    flac_exists = ASVSPOOF_EVAL_FLAC_DIR.exists()
    labels_exists = ASVSPOOF_EVAL_LABELS.exists()

    print(f"Eval directory found: {eval_exists}")
    print(f"FLAC directory found: {flac_exists}")
    print(f"Label file found    : {labels_exists}")
    print()

    if not eval_exists or not flac_exists or not labels_exists:
        print("Eval dataset path is not correct. Please check data/raw folder structure.")
        return

    audio_files = list(ASVSPOOF_EVAL_FLAC_DIR.glob("*.flac"))

    labels_df = pd.read_csv(
        ASVSPOOF_EVAL_LABELS,
        sep=r"\s+",
        header=None,
    )

    print(f"Number of audio files: {len(audio_files)}")
    print(f"Number of protocol rows: {len(labels_df)}")
    print()
    print("First 5 rows:")
    print(labels_df.head())
    print()

    # ASVspoof protocol files usually contain the audio file name in column 1.
    audio_file_column = labels_df.iloc[:, 1]

    audio_file_stems = {path.stem for path in audio_files}
    protocol_file_names = set(audio_file_column)

    missing_audio_files = protocol_file_names - audio_file_stems

    print(f"Missing audio files according to protocol: {len(missing_audio_files)}")

    if len(missing_audio_files) == 0:
        print("Eval dataset check passed successfully.")
    else:
        print("Some files listed in the eval protocol are missing from the flac folder.")
        print("Example missing files:")
        print(list(missing_audio_files)[:10])

    print()
    print("Protocol column count:", labels_df.shape[1])

    if labels_df.shape[1] >= 5:
        print()
        print("Possible label distribution from last column:")
        print(labels_df.iloc[:, -1].value_counts())


if __name__ == "__main__":
    main()