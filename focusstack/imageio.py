"""Wczytywanie i zapis obrazów (JPG, TIFF 8/16 bit).

Wewnętrznie wszystkie obrazy są trzymane jako float32 BGR w zakresie 0..1.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

from .errors import load_error, save_error

SUPPORTED_EXT = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


def is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXT


def load(path: str) -> tuple[np.ndarray, int]:
    """Zwraca (obraz float32 BGR 0..1, głębia bitowa źródła: 8 lub 16)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)  # obsługuje ścieżki z polskimi znakami
        img = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    except OSError as e:
        raise load_error(path, e) from e
    if img is None:
        raise load_error(path)
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0, 16
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0, 8
    return np.clip(img.astype(np.float32), 0, 1), 16


def load_gray_small(path: str, max_side: int) -> tuple[np.ndarray, float]:
    """Szybki podgląd w skali szarości do dopasowywania. Zwraca (obraz, skala)."""
    img, _ = load(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return gray, scale


def save(path: str, img: np.ndarray, bits: int = 8, exif_source: str | None = None,
         jpeg_quality: int = 95) -> None:
    try:
        _save(path, img, bits, exif_source, jpeg_quality)
    except Exception as e:
        raise save_error(path, e) from e


def _save(path: str, img: np.ndarray, bits: int, exif_source: str | None,
          jpeg_quality: int) -> None:
    """Zapisuje wynik. JPG zawsze 8 bit (z kopią EXIF i profilu ICC źródła), TIFF 8 lub 16 bit."""
    ext = os.path.splitext(path)[1].lower()
    img = np.clip(img, 0.0, 1.0)

    if ext in (".jpg", ".jpeg"):
        rgb = cv2.cvtColor((img * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_BGR2RGB)
        out = Image.fromarray(rgb)
        kwargs = {"quality": jpeg_quality, "subsampling": 0}
        if exif_source:
            try:
                with Image.open(exif_source) as src:
                    exif = src.getexif()
                    exif[0x0112] = 1  # orientacja: obraz jest już obrócony przy wczytaniu
                    kwargs["exif"] = exif.tobytes()
                    if src.info.get("icc_profile"):
                        kwargs["icc_profile"] = src.info["icc_profile"]
            except Exception:
                pass
        out.save(path, **kwargs)
        return

    if bits == 16:
        arr = (img * 65535.0 + 0.5).astype(np.uint16)
    else:
        arr = (img * 255.0 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(ext if ext in (".tif", ".tiff", ".png") else ".tif", arr)
    if not ok:
        raise OSError(f"kodowanie do formatu {ext} nie powiodło się")
    buf.tofile(path)
