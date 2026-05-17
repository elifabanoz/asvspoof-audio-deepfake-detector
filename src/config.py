from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Result directories
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Model directory
MODELS_DIR = ROOT_DIR / "models"

# Audio settings
SAMPLE_RATE = 16000
N_MFCC = 20

# Subset settings
N_BONAFIDE = 1000
N_SPOOF = 1000
RANDOM_STATE = 42

# ASVspoof expected paths
ASVSPOOF_TRAIN_DIR = RAW_DATA_DIR / "ASVspoof2019_LA_train"
ASVSPOOF_TRAIN_FLAC_DIR = ASVSPOOF_TRAIN_DIR / "flac"
ASVSPOOF_PROTOCOL_DIR = RAW_DATA_DIR / "ASVspoof2019_LA_cm_protocols"
ASVSPOOF_TRAIN_LABELS = ASVSPOOF_PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"

# Output files
SUBSET_CSV = PROCESSED_DATA_DIR / "subset_2000.csv"
FEATURES_NPZ = PROCESSED_DATA_DIR / "features.npz"

RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
MLP_MODEL_PATH = MODELS_DIR / "spoof_detector_mlp.pt"
MLP_SCALER_PATH = MODELS_DIR / "mlp_scaler.joblib"

METRICS_JSON = METRICS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"

# ASVspoof dev paths
ASVSPOOF_DEV_DIR = RAW_DATA_DIR / "ASVspoof2019_LA_dev"
ASVSPOOF_DEV_FLAC_DIR = ASVSPOOF_DEV_DIR / "flac"
ASVSPOOF_DEV_LABELS = ASVSPOOF_PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"

# Dev output files
DEV_SUBSET_CSV = PROCESSED_DATA_DIR / "dev_subset_2000.csv"
DEV_FEATURES_NPZ = PROCESSED_DATA_DIR / "dev_features.npz"