from pathlib import Path

import cv2

import base64

def decode_qr(qr_path: Path) -> bytes:
    """
    QR görselinden binary payload çıkarır.

    Args:
        qr_path: QR görselinin yolu.

    Returns:
        QR içerisinde bulunan binary veri.
    """

    image = cv2.imread(str(qr_path))

    if image is None:
        raise FileNotFoundError(
            f"{qr_path} bulunamadı."
        )

    detector = cv2.QRCodeDetector()

    data, points, _ = detector.detectAndDecode(image)

    if data == "":
        raise ValueError(
            "QR içerisinde veri bulunamadı."
        )

    return base64.b64decode(data)