from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from src.config import EMBEDDINGS_DIR

# Model yalnızca bir kez yüklensin
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0, det_size=(640, 640))


def extract_embedding(image_path: Path) -> np.ndarray:
    """
    Verilen fotoğraftan yüz embedding'i üretir.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"{image_path} bulunamadı.")

    faces = app.get(image)

    if len(faces) == 0:
        raise ValueError("Fotoğrafta yüz bulunamadı.")

    if len(faces) > 1:
        print("Uyarı: Birden fazla yüz bulundu. İlk yüz kullanılacak.")

    return faces[0].embedding


def save_embedding(embedding: np.ndarray, filename: str) -> Path:
    """
    Embedding'i .npy olarak kaydeder.
    """

    output_path = EMBEDDINGS_DIR / filename

    np.save(output_path, embedding)

    return output_path