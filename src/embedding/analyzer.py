import numpy as np


def analyze_embedding(embedding: np.ndarray) -> dict:
    """
    Embedding hakkında temel istatistikleri hesaplar.
    """

    return {
        "shape": embedding.shape,
        "dtype": str(embedding.dtype),
        "size": embedding.size,
        "bytes": embedding.nbytes,
        "min": float(embedding.min()),
        "max": float(embedding.max()),
        "mean": float(embedding.mean()),
        "std": float(embedding.std()),
        "l2_norm": float(np.linalg.norm(embedding)),
    }


def print_embedding_analysis(stats: dict) -> None:
    """
    Embedding analiz sonuçlarını ekrana yazdırır.
    """

    print("=" * 50)
    print("Embedding Analizi")
    print("=" * 50)

    print(f"Boyut           : {stats['shape']}")
    print(f"Veri tipi       : {stats['dtype']}")
    print(f"Toplam eleman   : {stats['size']}")
    print(f"Byte            : {stats['bytes']}")

    print(f"Min değer       : {stats['min']:.6f}")
    print(f"Max değer       : {stats['max']:.6f}")

    print(f"Ortalama        : {stats['mean']:.6f}")
    print(f"Std Sapma       : {stats['std']:.6f}")

    print(f"L2 Norm         : {stats['l2_norm']:.6f}")