import torch
import torch.nn as nn


class FaceGenerator(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(
                512,
                512 * 4 * 4
            ),
            nn.LeakyReLU(0.2)
        )

        self.decoder = nn.Sequential(

            # 4x4 -> 8x8
            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(
                32,
                3,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Tanh()
        )

    def forward(self, x):

        x = self.fc(x)

        x = x.view(
            -1,
            512,
            4,
            4
        )

        x = self.decoder(x)

        return x
