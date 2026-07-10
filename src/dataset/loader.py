import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import IMAGE_DIR, EMBEDDING_DIR


class FaceDataset(Dataset):

    def __init__(self):
        self.embedding_files = sorted(EMBEDDING_DIR.glob("*.npy"))

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, index):

        embedding_path = self.embedding_files[index]

        image_path = IMAGE_DIR / f"{embedding_path.stem}.jpg"

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        embedding = np.load(embedding_path).astype(np.float32)

        buffer = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Görüntü okunamadı: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image).permute(2, 0, 1)

        embedding = torch.from_numpy(embedding)

        return embedding, image