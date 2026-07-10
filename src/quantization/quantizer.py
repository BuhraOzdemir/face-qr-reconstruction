from pathlib import Path
import numpy as np
import struct

def quantize_embedding(embedding: np.ndarray):
    """
    Symmetric INT8 Quantization
    """

    scale = np.max(np.abs(embedding)) / 127.0

    quantized = np.round(embedding / scale).astype(np.int8)

    return quantized, scale


def dequantize_embedding(quantized: np.ndarray, scale: float):

    return quantized.astype(np.float32) * scale

def pack_embedding(quantized: np.ndarray, scale: float) -> bytes:
    """
    Binary format:
    [4 byte scale][512 byte embedding]
    """

    scale_bytes = struct.pack("<f", scale)

    embedding_bytes = quantized.tobytes()

    return scale_bytes + embedding_bytes


def unpack_embedding(data: bytes):

    scale = struct.unpack("<f", data[:4])[0]

    embedding = np.frombuffer(data[4:], dtype=np.int8)

    return embedding, scale