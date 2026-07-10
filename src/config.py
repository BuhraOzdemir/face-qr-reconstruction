"""
Path configuration for the face-qr project.

Detects Google Colab vs local (Windows) and builds all dataset /
output paths from a single DATA_ROOT so the same codebase runs
unchanged in both environments.
"""

import os
from pathlib import Path

# Project root (repository directory containing src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# True when running inside Google Colab
IS_COLAB = os.path.exists("/content")

# Data lives on Drive in Colab; next to the repo on Windows
if IS_COLAB:
    DATA_ROOT = Path("/content/drive/MyDrive/face_qr_project")
else:
    DATA_ROOT = PROJECT_ROOT

# -----------------------------
# Original dataset
# -----------------------------
DATASET_DIR = DATA_ROOT / "dataset"
CELEBA_DIR = DATASET_DIR / "celeba"
CELEBA_IMAGE_DIR = CELEBA_DIR / "img_align_celeba"

# -----------------------------
# Processed dataset
# -----------------------------
PROCESSED_DIR = DATA_ROOT / "data_processed"
IMAGE_DIR = PROCESSED_DIR / "images_128"
EMBEDDING_DIR = PROCESSED_DIR / "embeddings"

# -----------------------------
# Training / runtime outputs
# -----------------------------
OUTPUTS_DIR = DATA_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
GENERATED_DIR = OUTPUTS_DIR / "generated"
LOG_DIR = OUTPUTS_DIR / "logs"
QR_OUTPUT_DIR = OUTPUTS_DIR / "qr_output"

# Create directories if they do not exist yet
for _path in (
    DATASET_DIR,
    CELEBA_IMAGE_DIR,
    IMAGE_DIR,
    EMBEDDING_DIR,
    CHECKPOINT_DIR,
    GENERATED_DIR,
    LOG_DIR,
    QR_OUTPUT_DIR,
):
    _path.mkdir(parents=True, exist_ok=True)
