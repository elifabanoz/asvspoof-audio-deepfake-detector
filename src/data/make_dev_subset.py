import pandas as pd

from src.config import (
    ASVSPOOF_DEV_LABELS,
    DEV_SUBSET_CSV,
    N_BONAFIDE,
    N_SPOOF,
    RANDOM_STATE,
)


def main():
    print("=" * 60)
    print("Creating Balanced ASVspoof Dev Subset")
    print("=" * 60)

    if not ASVSPOOF_DEV_LABELS.exists():
        raise FileNotFoundError(f"Dev label file not found: {ASVSPOOF_DEV_LABELS}")

    labels_df = pd.read_csv(
        ASVSPOOF_DEV_LABELS,
        sep=r"\s+",
        header=None,
        names=["speaker_id", "audio_file", "unused", "system_id", "label"],
    )

    print("Original dev label distribution:")
    print(labels_df["label"].value_counts())
    print()

    bonafide_df = labels_df[labels_df["label"] == "bonafide"]
    spoof_df = labels_df[labels_df["label"] == "spoof"]

    if len(bonafide_df) < N_BONAFIDE:
        raise ValueError(
            f"Not enough bonafide samples. Requested {N_BONAFIDE}, found {len(bonafide_df)}"
        )

    if len(spoof_df) < N_SPOOF:
        raise ValueError(
            f"Not enough spoof samples. Requested {N_SPOOF}, found {len(spoof_df)}"
        )

    bonafide_subset = bonafide_df.sample(
        n=N_BONAFIDE,
        random_state=RANDOM_STATE,
    )

    spoof_subset = spoof_df.sample(
        n=N_SPOOF,
        random_state=RANDOM_STATE,
    )

    subset_df = pd.concat([bonafide_subset, spoof_subset], axis=0)

    subset_df = subset_df.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    subset_df["label_id"] = subset_df["label"].map(
        {
            "bonafide": 0,
            "spoof": 1,
        }
    )

    DEV_SUBSET_CSV.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(DEV_SUBSET_CSV, index=False)

    print("Balanced dev subset created successfully.")
    print(f"Saved to: {DEV_SUBSET_CSV}")
    print()
    print("Dev subset label distribution:")
    print(subset_df["label"].value_counts())
    print()
    print("First 5 rows:")
    print(subset_df.head())


if __name__ == "__main__":
    main()