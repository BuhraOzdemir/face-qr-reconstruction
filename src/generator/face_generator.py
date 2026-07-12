import torch
import torch.nn as nn


class FaceGenerator(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super(FaceGenerator, self).__init__()

        self.fc = nn.Linear(embedding_dim, 512 * 4 * 4)

        self.decoder = nn.Sequential(
            # 512 x 4 x 4 -> 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 256 x 8 x 8 -> 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 x 16 x 16 -> 64 x 32 x 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 32 x 32 -> 32 x 64 x 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 x 64 x 64 -> 3 x 128 x 128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        x = self.decoder(x)
        return x
