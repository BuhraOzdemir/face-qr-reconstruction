import torch

from src.generator.face_generator import FaceGenerator


model = FaceGenerator()


embedding = torch.randn(
    1,
    512
)


output = model(embedding)


print("Input:")
print(embedding.shape)


print("\nGenerated Image:")
print(output.shape)