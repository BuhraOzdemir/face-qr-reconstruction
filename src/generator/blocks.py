import torch.nn as nn


class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)