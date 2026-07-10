from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from insightface.app import FaceAnalysis


# -----------------------------
# Paths
# -----------------------------

CELEBA_DIR = Path(
    "dataset/celeba/img_align_celeba"
)

OUTPUT_IMAGE_DIR = Path(
    "data_processed/images_128"
)

OUTPUT_EMBEDDING_DIR = Path(
    "data_processed/embeddings"
)


OUTPUT_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_EMBEDDING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------
# MobileFaceNet
# -----------------------------

app = FaceAnalysis(
    name="buffalo_s"
)

app.prepare(
    ctx_id=0,
    det_size=(640,640)
)


# -----------------------------
# Dataset preparation
# -----------------------------

def prepare_dataset(limit=100):

    images = sorted(
    CELEBA_DIR.rglob("*.jpg")
    )


    images = images[:limit]


    print(
        f"Toplam işlenecek görüntü: {len(images)}"
    )


    for img_path in tqdm(images):

        image = cv2.imread(
            str(img_path)
        )


        if image is None:
            continue


        faces = app.get(image)


        if len(faces) == 0:
            continue


        face = faces[0]


        embedding = face.embedding


        # 128x128 resize
        resized = cv2.resize(
            image,
            (128,128)
        )


        # kayıt isimleri
        name = img_path.stem


        cv2.imwrite(
            str(
                OUTPUT_IMAGE_DIR /
                f"{name}.jpg"
            ),
            resized
        )


        np.save(
            OUTPUT_EMBEDDING_DIR /
            f"{name}.npy",
            embedding
        )


    print(
        "Dataset hazırlama tamamlandı."
    )


if __name__ == "__main__":

    prepare_dataset(
        limit=10000
    )