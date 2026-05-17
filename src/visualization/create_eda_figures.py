import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import librosa.display

from src.config import (
    ASVSPOOF_TRAIN_FLAC_DIR,
    FIGURES_DIR,
    SAMPLE_RATE,
    SUBSET_CSV,
)

from src.utils import get_audio_path, load_subset


def load_audio(audio_file_name: str):
    audio_path = get_audio_path(ASVSPOOF_TRAIN_FLAC_DIR, audio_file_name)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return audio, sr, audio_path


def plot_label_distribution(subset_df: pd.DataFrame):
    label_counts = subset_df["label"].value_counts()

    plt.figure(figsize=(6, 4))
    plt.bar(label_counts.index, label_counts.values)
    plt.xlabel("Label")
    plt.ylabel("Number of Samples")
    plt.title("Balanced Subset Label Distribution")
    plt.tight_layout()

    output_path = FIGURES_DIR / "label_distribution_subset.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved label distribution figure to: {output_path}")


def plot_waveforms(real_audio, spoof_audio, sr, real_name, spoof_name):
    time_real = np.arange(len(real_audio)) / sr
    time_spoof = np.arange(len(spoof_audio)) / sr

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_real, real_audio)
    plt.title(f"Bonafide Waveform: {real_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(time_spoof, spoof_audio)
    plt.title(f"Spoof Waveform: {spoof_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.tight_layout()

    output_path = FIGURES_DIR / "bonafide_vs_spoof_waveform.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved waveform comparison figure to: {output_path}")


def plot_mel_spectrograms(real_audio, spoof_audio, sr, real_name, spoof_name):
    real_mel = librosa.feature.melspectrogram(
        y=real_audio,
        sr=sr,
        n_mels=128,
    )
    spoof_mel = librosa.feature.melspectrogram(
        y=spoof_audio,
        sr=sr,
        n_mels=128,
    )

    real_mel_db = librosa.power_to_db(real_mel, ref=np.max)
    spoof_mel_db = librosa.power_to_db(spoof_mel, ref=np.max)

    plt.figure(figsize=(12, 7))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(
        real_mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Bonafide Mel Spectrogram: {real_name}")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(
        spoof_mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spoof Mel Spectrogram: {spoof_name}")

    plt.tight_layout()

    output_path = FIGURES_DIR / "bonafide_vs_spoof_mel_spectrogram.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved mel spectrogram comparison figure to: {output_path}")


def main():
    print("=" * 60)
    print("Creating EDA Figures")
    print("=" * 60)

    subset_df = load_subset(SUBSET_CSV)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_label_distribution(subset_df)

    bonafide_row = subset_df[subset_df["label"] == "bonafide"].iloc[0]
    spoof_row = subset_df[subset_df["label"] == "spoof"].iloc[0]

    real_audio, sr, real_path = load_audio(bonafide_row["audio_file"])
    spoof_audio, _, spoof_path = load_audio(spoof_row["audio_file"])

    print()
    print("Selected audio samples:")
    print(f"Bonafide: {bonafide_row['audio_file']} -> {real_path}")
    print(f"Spoof   : {spoof_row['audio_file']} -> {spoof_path}")
    print()

    plot_waveforms(
        real_audio,
        spoof_audio,
        sr,
        bonafide_row["audio_file"],
        spoof_row["audio_file"],
    )

    plot_mel_spectrograms(
        real_audio,
        spoof_audio,
        sr,
        bonafide_row["audio_file"],
        spoof_row["audio_file"],
    )

    print()
    print("EDA figures created successfully.")


if __name__ == "__main__":
    main()