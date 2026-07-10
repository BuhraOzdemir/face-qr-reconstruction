import torch
import torch.nn as nn


class MappingNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512)

        )


    def forward(self, x):

        return self.network(x)