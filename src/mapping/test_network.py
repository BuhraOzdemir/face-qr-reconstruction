import torch

from src.mapping.network import MappingNetwork


model = MappingNetwork()

x = torch.randn(1,512)

output = model(x)


print("Input :", x.shape)
print("Output:", output.shape)