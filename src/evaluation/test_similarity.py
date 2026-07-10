from pathlib import Path

import numpy as np

from src.qr.decoder import decode_qr
from src.quantization.quantizer import (
    unpack_embedding,
    dequantize_embedding,
)
from src.evaluation.similarity import (
    cosine_similarity,
)


# Orijinal embedding
original = np.load(
    "embeddings/andy_Samberg.npy"
)


# QR oku
payload = decode_qr(
    Path("qr_output/andy_Samberg.png")
)


# INT8 + scale ayır
quantized, scale = unpack_embedding(
    payload
)


# Float32'e geri dön
restored = dequantize_embedding(
    quantized,
    scale,
)


similarity = cosine_similarity(
    original,
    restored,
)


print("=" * 50)
print("Embedding Similarity Test")
print("=" * 50)

print(f"Cosine Similarity : {similarity:.8f}")