from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.config import GENERATED_DIR


def save_comparison(
    generated: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
):

    generated = generated.detach().cpu()
    target = target.detach().cpu()

    generated = generated.permute(1, 2, 0).numpy()
    target = target.permute(1, 2, 0).numpy()

    generated = generated.clip(0, 1)
    target = target.clip(0, 1)

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(target)
    plt.title("Target")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(generated)
    plt.title("Generated")
    plt.axis("off")

    output_path = GENERATED_DIR / f"epoch_{epoch:03d}.png"

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()