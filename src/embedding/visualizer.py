from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_embedding_distribution(
    embedding: np.ndarray,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """
    Embedding değerlerinin dağılımını histogram olarak çizer.

    Args:
        embedding: Analiz edilecek embedding.
        save_path: Grafik kaydedilecekse dosya yolu.
        show: Grafik ekranda gösterilsin mi?
    """

    plt.figure(figsize=(8, 5))

    plt.hist(embedding, bins=40)

    plt.title("Distribution of Face Embedding Values")
    plt.xlabel("Embedding Value")
    plt.ylabel("Frequency")

    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()