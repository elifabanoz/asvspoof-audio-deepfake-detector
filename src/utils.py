import json
from pathlib import Path

import pandas as pd


LABEL_TO_ID = {
    "bonafide": 0,
    "spoof": 1,
}

ID_TO_LABEL = {
    0: "bonafide",
    1: "spoof",
}


def load_subset(subset_csv: Path) -> pd.DataFrame:
    if not subset_csv.exists():
        raise FileNotFoundError(
            f"Subset file not found: {subset_csv}\n"
            "Run this first: python src\\make_subset.py"
        )

    df = pd.read_csv(subset_csv)

    required_columns = {"audio_file", "label", "label_id"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in subset CSV: {missing_columns}")

    return df


def get_audio_path(flac_dir: Path, audio_file: str) -> Path:
    return flac_dir / f"{audio_file}.flac"


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)