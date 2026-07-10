import torch

from src.model.face_reconstruction import FaceReconstructionModel


model = FaceReconstructionModel()


embedding = torch.randn(
    1,
    512
)


output = model(embedding)


print("Embedding:")
print(embedding.shape)


print("\nGenerated Image:")
print(output.shape)