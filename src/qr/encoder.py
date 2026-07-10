from pathlib import Path

import qrcode
from torch import qr

from src.config import QR_OUTPUT_DIR
import base64


def generate_qr(payload: bytes, filename: str) -> Path:
    """
    Binary payload'dan QR kodu oluşturur ve kaydeder.

    Args:
        payload: QR içine yazılacak binary veri.
        filename: Kaydedilecek QR dosyasının adı.

    Returns:
        Oluşturulan QR dosyasının yolu.
    """

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )

    encoded_payload = base64.b64encode(payload).decode("ascii")

    qr.add_data(encoded_payload)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    output_path = QR_OUTPUT_DIR / filename

    img.save(output_path)

    return output_path