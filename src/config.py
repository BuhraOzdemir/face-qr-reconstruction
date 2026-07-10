from pathlib import Path

# -----------------------------
# Project Root
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------
# Original Dataset
# -----------------------------

DATASET_DIR = PROJECT_ROOT / "dataset"

CELEBA_DIR = DATASET_DIR / "celeba"

CELEBA_IMAGE_DIR = CELEBA_DIR / "img_align_celeba"

# -----------------------------
# Processed Dataset
# -----------------------------

PROCESSED_DIR = PROJECT_ROOT / "data_processed"

IMAGE_DIR = PROCESSED_DIR / "images_128"

EMBEDDING_DIR = PROCESSED_DIR / "embeddings"

# -----------------------------
# Previous folders
# -----------------------------

OUTPUT_DIR = PROJECT_ROOT / "embeddings"

QR_OUTPUT_DIR = PROJECT_ROOT / "qr_output"

# -----------------------------
# Training Outputs
# -----------------------------

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"

GENERATED_DIR = OUTPUTS_DIR / "generated"

LOG_DIR = OUTPUTS_DIR / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Outputs
# -----------------------------

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"

GENERATED_DIR = OUTPUTS_DIR / "generated"

LOG_DIR = OUTPUTS_DIR / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)