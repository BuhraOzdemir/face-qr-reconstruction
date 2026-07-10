import torch.nn as nn

from src.mapping.network import MappingNetwork
from src.generator.face_generator import FaceGenerator


class FaceReconstructionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.mapping = MappingNetwork()

        self.generator = FaceGenerator()


    def forward(self, embedding):

        latent = self.mapping(embedding)

        image = self.generator(latent)

        return image