import csv

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import IMAGE_DIR, EMBEDDING_DIR, SPLIT_FILE


class FaceDataset(Dataset):

    def __init__(self, split: str = "train"):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val' or 'test', got '{split}'"

        self.split = split
        self.samples = self._load_split()

    def _load_split(self):
        samples = []
        with open(SPLIT_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"].strip() == self.split:
                    samples.append(row["image_id"].strip())
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        stem = self.samples[index]

        embedding_path = EMBEDDING_DIR / f"{stem}.npy"
        image_path = IMAGE_DIR / f"{stem}.jpg"

        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding bulunamadı: {embedding_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")

        embedding = np.load(embedding_path).astype(np.float32)

        buffer = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Görüntü okunamadı: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize to [-1, 1] to match Tanh generator output
        image = image.astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        embedding = torch.from_numpy(embedding)

        return embedding, image
