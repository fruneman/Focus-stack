"""Generator syntetycznej serii makro o znanej "prawdzie" (do testów jakości).

Scena: teksturowane tło o głębokości rosnącej od lewej do prawej + cienkie,
ciemne "czułki" na pierwszym planie (najbliżej obiektywu). Każda klatka ma
ostrość na innej głębokości, a do tego symulowane "oddychanie" obiektywu
(zmiana skali), przesunięcie i minimalny obrót.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def _texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((h, w, 3), np.float32)
    for scale in (2, 6, 18, 50):
        noise = rng.random((h // scale + 2, w // scale + 2, 3)).astype(np.float32)
        img += cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC) / 4
    # drobne kropki i kreski - "łuski" / "włoski"
    for _ in range(4000):
        x, y = rng.integers(0, w), rng.integers(0, h)
        c = tuple(float(v) for v in rng.random(3))
        cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 4)), c, -1, cv2.LINE_AA)
    return np.clip(img * 0.8 + 0.1, 0, 1)


def _antennae(h: int, w: int) -> np.ndarray:
    alpha = np.zeros((h, w), np.float32)
    for k, (x0, y0) in enumerate([(0.35, 0.95), (0.45, 0.95), (0.6, 0.9)]):
        pts = []
        for t in np.linspace(0, 1, 200):
            x = (x0 + 0.25 * t * (1 if k % 2 else -1) + 0.03 * np.sin(6 * t)) * w
            y = (y0 - 0.8 * t) * h
            pts.append((x, y))
        pts = np.int32(np.array(pts) * 16)
        cv2.polylines(alpha, [pts], False, 1.0, 3, cv2.LINE_AA, shift=4)
        # "włoski" na czułkach
        for i in range(0, 200, 8):
            px, py = pts[i] / 16
            cv2.line(alpha, (int(px), int(py)), (int(px + 12), int(py - 8)), 1.0, 1, cv2.LINE_AA)
    return alpha


def disc_blur(img: np.ndarray, r: float) -> np.ndarray:
    """Rozmycie krążkiem o ostrych brzegach - tak jak bokeh prawdziwego obiektywu."""
    if r < 0.5:
        return img
    k = int(np.ceil(r)) * 2 + 1
    yy, xx = np.mgrid[:k, :k] - k // 2
    d = np.sqrt(xx ** 2 + yy ** 2)
    ker = np.clip(r + 0.5 - d, 0, 1).astype(np.float32)  # krążek z antyaliasingiem brzegu
    ker /= ker.sum()
    return cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REFLECT)


def make_series(out_dir: str, n: int = 12, w: int = 1800, h: int = 1200, seed: int = 1,
                breathing: float = 0.003, jitter: float = 4.0) -> dict:
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    bg = _texture(h, w, rng)
    alpha = _antennae(h, w)
    fg_color = np.array([0.05, 0.08, 0.12], np.float32)

    # głębokości: tło 0.3 (lewo) .. 1.0 (prawo), czułki 0.0
    bands = 7
    band_depth = np.linspace(0.3, 1.0, bands)
    xs = np.linspace(0, bands, w, endpoint=False).astype(int)
    truth = bg * (1 - alpha[..., None]) + fg_color * alpha[..., None]

    focus = np.linspace(0.0, 1.0, n)
    ref = n // 2
    mats = []
    for i, f in enumerate(focus):
        # tło: rozmycie zależne od odległości głębokości od płaszczyzny ostrości
        frame_bg = np.zeros_like(bg)
        for b, d in enumerate(band_depth):
            r = abs(d - f) * 20
            blurred = disc_blur(bg, r)
            mask = (xs == b)[None, :, None]
            frame_bg = np.where(mask, blurred, frame_bg)
        r_fg = abs(0.0 - f) * 20
        a = disc_blur(alpha, r_fg)
        frame = frame_bg * (1 - a[..., None]) + fg_color * a[..., None]
        frame *= 1.0 + 0.02 * (i - ref) / n  # lekka zmiana ekspozycji

        # ruch aparatu i oddychanie obiektywu
        s = 1.0 + breathing * (i - ref)
        ang = np.deg2rad(rng.normal(0, 0.05))
        tx, ty = rng.normal(0, jitter, 2) if i != ref else (0.0, 0.0)
        if i == ref:
            s, ang = 1.0, 0.0
        c, sn = s * np.cos(ang), s * np.sin(ang)
        cx, cy = w / 2, h / 2
        m = np.array([[c, -sn, cx - c * cx + sn * cy + tx],
                      [sn, c, cy - sn * cx - c * cy + ty]], np.float32)
        mats.append(m)
        frame = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT)
        frame += rng.normal(0, 0.008, frame.shape).astype(np.float32)  # szum
        u8 = np.clip(frame * 255 + 0.5, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:03d}.jpg"), u8, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {"truth": truth, "alpha": alpha, "matrices": mats, "ref": ref}
