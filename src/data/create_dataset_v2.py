"""
CelebA preprocessing pipeline v2.

Reads images directly from the archive ZIP (no full extraction),
detects faces with InsightFace buffalo_s, and saves:
  - 128x128 JPEG  -> data_processed_v2/images_128/{image_id}.jpg
  - 512-d embedding -> data_processed_v2/embeddings/{image_id}.npy

Partition mapping  (list_eval_partition.csv):
  0 -> train   (up to 20 000 samples)
  1 -> val     (up to  3 000 samples)
  2 -> test    (up to  3 000 samples)

Designed to run inside Google Colab with Drive mounted.
"""

import io
import logging
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ZIP_PATH = Path("/content/drive/MyDrive/Staj2026/archive.zip")

# CSV shipped inside the cloned repo
CSV_PATH = Path("/content/face-qr-reconstruction/list_eval_partition.csv")

# Output root on Drive so processed data survives Colab session resets
DATA_ROOT = Path("/content/drive/MyDrive/Staj2026/face_qr_project")
OUT_IMAGES = DATA_ROOT / "data_processed_v2" / "images_128"
OUT_EMBEDDINGS = DATA_ROOT / "data_processed_v2" / "embeddings"

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_EMBEDDINGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Per-split sample limits
# ---------------------------------------------------------------------------
SPLIT_LIMITS = {0: 20_000, 1: 3_000, 2: 3_000}
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}

# Prefix used inside the ZIP (adjust if your archive differs)
ZIP_IMAGE_PREFIX = "img_align_celeba/img_align_celeba/"

TARGET_SIZE = (128, 128)
EMBEDDING_DIM = 512

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InsightFace model
# ---------------------------------------------------------------------------
def load_model() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_s")
    app.prepare(ctx_id=0, det_size=(640, 640))
    log.info("InsightFace buffalo_s loaded (ctx_id=0).")
    return app


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def read_image_from_zip(zf: zipfile.ZipFile, zip_path: str) -> np.ndarray | None:
    """Return a BGR numpy array read from a zip member, or None on error."""
    try:
        data = zf.read(zip_path)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        log.warning("Cannot read %s from zip: %s", zip_path, exc)
        return None


def extract_embedding(app: FaceAnalysis, img_bgr: np.ndarray) -> np.ndarray | None:
    """Run face detection + embedding. Returns 512-d vector or None."""
    faces = app.get(img_bgr)
    if not faces:
        return None
    # Use the face with the largest bounding-box area when multiple faces found
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = face.embedding
    if emb is None or len(emb) != EMBEDDING_DIM:
        return None
    return emb.astype(np.float32)


def save_sample(image_id: str, img_bgr: np.ndarray, emb: np.ndarray) -> None:
    resized = cv2.resize(img_bgr, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT_IMAGES / f"{image_id}.jpg"), resized)
    np.save(str(OUT_EMBEDDINGS / f"{image_id}.npy"), emb)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run() -> None:
    log.info("Reading partition CSV: %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Build per-split lists  {partition -> [image_id, ...]}
    # Reproducible random sampling (random_state=42) instead of head-N.
    split_ids: dict[int, list[str]] = {}
    for partition, limit in SPLIT_LIMITS.items():
        pool = df.loc[df["partition"] == partition, "image_id"]
        ids = pool.sample(n=min(limit, len(pool)), random_state=42).tolist()
        split_ids[partition] = ids
        log.info(
            "Split %-5s : %d requested / %d available",
            SPLIT_NAMES[partition],
            limit,
            len(pool),
        )

    app = load_model()

    error_log: list[str] = []
    split_records: list[dict] = []

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for partition, ids in split_ids.items():
            split_name = SPLIT_NAMES[partition]
            saved = skipped = 0

            log.info("Processing split: %s  (%d images)", split_name, len(ids))
            pbar = tqdm(ids, desc=split_name, unit="img", dynamic_ncols=True)

            for idx, image_id in enumerate(pbar):
                # Skip already processed samples (allows resuming)
                if (OUT_EMBEDDINGS / f"{image_id}.npy").exists():
                    saved += 1
                    continue

                zip_member = ZIP_IMAGE_PREFIX + image_id
                img = read_image_from_zip(zf, zip_member)

                if img is None:
                    msg = f"[{split_name}] {image_id}: cannot decode image"
                    error_log.append(msg)
                    skipped += 1
                    continue

                emb = extract_embedding(app, img)
                if emb is None:
                    msg = f"[{split_name}] {image_id}: no face detected"
                    error_log.append(msg)
                    skipped += 1
                    continue

                save_sample(image_id, img, emb)
                split_records.append({"image_id": image_id, "split": split_name})
                saved += 1

                if (idx + 1) % 100 == 0:
                    log.info(
                        "[%s] %d/%d  saved=%d  skipped=%d",
                        split_name, idx + 1, len(ids), saved, skipped,
                    )

            log.info(
                "Split %s done — saved: %d  skipped: %d", split_name, saved, skipped
            )

    # Write split.csv
    split_csv_path = DATA_ROOT / "data_processed_v2" / "split.csv"
    pd.DataFrame(split_records).to_csv(split_csv_path, index=False)
    log.info("Split CSV written to %s  (%d rows)", split_csv_path, len(split_records))

    # Write error log next to the outputs
    if error_log:
        err_path = DATA_ROOT / "data_processed_v2" / "errors.log"
        err_path.write_text("\n".join(error_log), encoding="utf-8")
        log.info("Error log written to %s  (%d entries)", err_path, len(error_log))
    else:
        log.info("No errors.")

    log.info(
        "Done.  images -> %s   embeddings -> %s", OUT_IMAGES, OUT_EMBEDDINGS
    )


if __name__ == "__main__":
    run()
