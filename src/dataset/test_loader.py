import torch
from torch.utils.data import DataLoader

from src.dataset.loader import FaceDataset

dataset = FaceDataset()

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
)

embeddings, images = next(iter(loader))

print("=" * 50)

print("Batch Embeddings :", embeddings.shape)

print("Batch Images     :", images.shape)

print("=" * 50)