import numpy as np


def cosine_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> float:
    """
    İki embedding arasındaki cosine similarity değerini hesaplar.
    """

    dot_product = np.dot(
        embedding1,
        embedding2
    )

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    return float(
        dot_product / (norm1 * norm2)
    )