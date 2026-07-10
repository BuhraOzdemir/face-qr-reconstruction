from pathlib import Path

from src.embedding.extractor import (
    extract_embedding,
    save_embedding,
)
from src.embedding.analyzer import (
    analyze_embedding,
    print_embedding_analysis,
)
from src.embedding.visualizer import (
    plot_embedding_distribution,
)
from src.quantization.quantizer import (
    quantize_embedding,
    pack_embedding,
)
from src.qr.encoder import (
    generate_qr,
)


def run_pipeline(image_path: Path) -> None:
    """
    Face QR pipeline'ını çalıştırır.
    """

    image_name = image_path.stem

    print("=" * 60)
    print("Face QR Pipeline")
    print("=" * 60)

    # 1. Embedding üret
    embedding = extract_embedding(image_path)

    # 2. Embedding'i kaydet
    embedding_path = save_embedding(
        embedding,
        f"{image_name}.npy",
    )

    print(f"Embedding kaydedildi -> {embedding_path}")

    # 3. Analiz
    stats = analyze_embedding(embedding)
    print_embedding_analysis(stats)

    # 4. Histogram
    plot_embedding_distribution(
        embedding,
        show=True,
    )

    # 5. Quantization
    quantized, scale = quantize_embedding(embedding)

    # 6. Binary payload
    payload = pack_embedding(
        quantized,
        scale,
    )

    # 7. QR oluştur
    qr_path = generate_qr(
        payload,
        f"{image_name}.png",
    )

    print(f"\nQR kaydedildi -> {qr_path}")

    print("\nPipeline başarıyla tamamlandı.")
original_embedding = np.load(
    EMBEDDING_DIR / "000001.npy"
)


score = calculate_identity_score(
    original_embedding,
    image
)


print(
    "Identity Score:",
    score
)

if __name__ == "__main__":

    IMAGE_PATH = Path("images") / "andy_Samberg.jpg"

    run_pipeline(IMAGE_PATH)