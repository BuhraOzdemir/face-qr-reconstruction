import torch

from src.training.losses import ReconstructionLoss

criterion = ReconstructionLoss()

prediction = torch.rand(4, 3, 128, 128)

target = torch.rand(4, 3, 128, 128)

loss = criterion(prediction, target)

print("=" * 50)
print("Loss Test")
print("=" * 50)
print(loss.item())