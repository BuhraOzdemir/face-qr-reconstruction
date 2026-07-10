from pathlib import Path

from src.qr.decoder import decode_qr
from src.quantization.quantizer import unpack_embedding


QR_PATH = Path(
    "qr_output/andy_Samberg.png"
)


payload = decode_qr(QR_PATH)

print("Payload boyutu:", len(payload))


embedding_int8, scale = unpack_embedding(
    payload
)

print("Embedding boyutu:", embedding_int8.shape)
print("Scale:", scale)