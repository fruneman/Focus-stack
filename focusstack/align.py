"""Dopasowanie (align) serii zdjęć do focus stackingu.

Strategia:
  * Klatką odniesienia jest środkowa klatka serii.
  * Każdą klatkę dopasowujemy do jej już dopasowanego sąsiada (sąsiednie
    klatki mają najbardziej podobną ostrość, więc dopasowanie jest pewne),
    a przekształcenia składamy w łańcuch aż do klatki odniesienia.
  * Pojedyncze dopasowanie: najpierw punkty charakterystyczne SIFT + RANSAC
    (zgrubnie, odporne na duże przesunięcia), potem dokładne doprecyzowanie
    algorytmem ECC (subpikselowo). Model ruchu: przesunięcie + obrót + skala
    (skala koryguje "oddychanie" obiektywu przy zmianie ostrości).
  * Na koniec wyrównanie jasności (osobno dla każdego kanału koloru).

Konwencja: macierz M (3x3) mapuje współrzędne klatki odniesienia na
współrzędne danej klatki. Warp wykonujemy z flagą WARP_INVERSE_MAP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np

from . import imageio

ProgressFn = Callable[[float, str], None]


@dataclass
class Alignment:
    size: tuple[int, int]                 # (szerokość, wysokość) pełnej rozdzielczości
    ref_index: int
    matrices: list[np.ndarray]            # 3x3, ref -> klatka, pełna rozdzielczość
    gains: list[np.ndarray] = field(default_factory=list)  # (3,) mnożnik BGR
    crop: tuple[int, int, int, int] | None = None          # x, y, w, h wspólnego obszaru
    warnings: list[str] = field(default_factory=list)

    def warp(self, img: np.ndarray, index: int) -> np.ndarray:
        w, h = self.size
        m = self.matrices[index][:2]
        out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)
        if self.gains:
            out *= self.gains[index][None, None, :]
        return out


def _to_similarity(m: np.ndarray) -> np.ndarray:
    """Rzutuje macierz afiniczną 2x3 na podobieństwo (bez ścinania / niejednorodnej skali)."""
    a, b = m[0, 0], m[0, 1]
    c, d = m[1, 0], m[1, 1]
    s = 0.5 * (np.hypot(a, c) + np.hypot(b, d))
    ang = np.arctan2(c - b, a + d)
    out = m.copy()
    out[0, 0] = s * np.cos(ang)
    out[0, 1] = -s * np.sin(ang)
    out[1, 0] = s * np.sin(ang)
    out[1, 1] = s * np.cos(ang)
    return out


def _prep(gray: np.ndarray) -> np.ndarray:
    """Normalizacja kontrastu (CLAHE) - pomaga przy różnicach ostrości i jasności."""
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(u8)


class _Features:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=5000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.cache: dict[int, tuple] = {}

    def get(self, idx: int, img_u8: np.ndarray):
        if idx not in self.cache:
            self.cache[idx] = self.sift.detectAndCompute(img_u8, None)
        return self.cache[idx]


def _estimate_pair(feat: _Features, ia: int, a: np.ndarray, ib: int, b: np.ndarray
                   ) -> tuple[np.ndarray, str | None]:
    """Szacuje macierz 2x3 mapującą współrzędne obrazu A na obraz B (mała skala)."""
    a8, b8 = _prep(a), _prep(b)
    init = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    warn = None

    # 1) Zgrubnie: SIFT + RANSAC
    ka, da = feat.get(ia, a8)
    kb, db = feat.get(ib, b8)
    if da is not None and db is not None and len(ka) >= 8 and len(kb) >= 8:
        matches = feat.matcher.knnMatch(da, db, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) >= 8:
            pa = np.float32([ka[m.queryIdx].pt for m in good])
            pb = np.float32([kb[m.trainIdx].pt for m in good])
            m, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                                 ransacReprojThreshold=2.0, maxIters=5000,
                                                 confidence=0.999)
            if m is not None and inl is not None and inl.sum() >= 8:
                init = m.astype(np.float32)

    # 2) Dokładnie: ECC na lekko rozmytych obrazach
    af = cv2.GaussianBlur(a8.astype(np.float32) / 255.0, (0, 0), 1.0)
    bf = cv2.GaussianBlur(b8.astype(np.float32) / 255.0, (0, 0), 1.0)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, m = cv2.findTransformECC(af, bf, init.copy(), cv2.MOTION_AFFINE, criteria, None, 5)
        m = _to_similarity(m)
        # Zabezpieczenie: ECC nie może "uciec" daleko od wyniku SIFT
        drift = np.abs(m[:, 2] - init[:, 2]).max()
        scale_diff = abs(np.hypot(m[0, 0], m[1, 0]) - np.hypot(init[0, 0], init[1, 0]))
        if drift > 0.02 * max(a.shape) or scale_diff > 0.02:
            warn = "ECC odrzucone, użyto dopasowania punktów"
            m = init
    except cv2.error:
        warn = "ECC nie zbiegło się, użyto dopasowania punktów"
        m = init
    return m.astype(np.float64), warn


def _gain(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Mnożniki BGR wyrównujące średnią jasność do klatki odniesienia (środek kadru)."""
    h, w = ref.shape[:2]
    y0, y1, x0, x1 = h // 6, h - h // 6, w // 6, w - w // 6
    r = ref[y0:y1, x0:x1].reshape(-1, 3).mean(0)
    i = img[y0:y1, x0:x1].reshape(-1, 3).mean(0)
    g = r / np.maximum(i, 1e-6)
    return np.clip(g, 0.7, 1.4).astype(np.float32)


def _common_crop(matrices: Sequence[np.ndarray], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Prostokąt (w układzie odniesienia) pokryty przez wszystkie klatki."""
    w, h = size
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
    left, top, right, bottom = 0.0, 0.0, float(w), float(h)
    for m in matrices:
        p = np.linalg.inv(m) @ corners  # rogi klatki w układzie odniesienia
        left = max(left, p[0, 0], p[0, 3])
        right = min(right, p[0, 1], p[0, 2])
        top = max(top, p[1, 0], p[1, 1])
        bottom = min(bottom, p[1, 2], p[1, 3])
    x0, y0 = int(np.ceil(left)), int(np.ceil(top))
    x1, y1 = int(np.floor(right)), int(np.floor(bottom))
    if x1 - x0 < w // 2 or y1 - y0 < h // 2:  # coś poszło nie tak - nie przycinamy
        return 0, 0, w, h
    return x0, y0, x1 - x0, y1 - y0


def align(paths: Sequence[str], work_size: int = 2000,
          progress: ProgressFn | None = None) -> Alignment:
    n = len(paths)
    if n == 0:
        raise ValueError("Brak zdjęć")
    progress = progress or (lambda f, m: None)

    smalls, thumbs = [], []
    full_size = None
    s = 1.0
    for i, p in enumerate(paths):
        progress(0.3 * i / n, f"Wczytywanie {i + 1}/{n}")
        img, _ = imageio.load(p)
        h, w = img.shape[:2]
        if full_size is None:
            full_size = (w, h)
            s = min(1.0, work_size / max(w, h))
        elif (w, h) != full_size:
            raise ValueError(f"Zdjęcie ma inny rozmiar niż pozostałe: {p}")
        small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        smalls.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        t = 400 / max(w, h)
        thumbs.append(cv2.resize(img, (round(w * t), round(h * t)), interpolation=cv2.INTER_AREA))
        del img

    ref = n // 2
    eye = np.eye(3)
    small_m: list[np.ndarray] = [eye.copy() for _ in range(n)]
    warnings: list[str] = []
    feat = _Features()

    order = [(i, i + 1) for i in range(ref, n - 1)] + [(i, i - 1) for i in range(ref, 0, -1)]
    for k, (a, b) in enumerate(order):
        progress(0.3 + 0.6 * k / max(1, len(order)), f"Dopasowywanie {k + 1}/{len(order)}")
        m, warn = _estimate_pair(feat, a, smalls[a], b, smalls[b])
        if warn:
            warnings.append(f"Klatka {b + 1}: {warn}")
        m3 = np.vstack([m, [0, 0, 1]])
        small_m[b] = m3 @ small_m[a]

    # Przeliczenie na pełną rozdzielczość: M_full = S^-1 · M_small · S
    sm = np.diag([s, s, 1.0])
    sm_inv = np.diag([1 / s, 1 / s, 1.0])
    matrices = [sm_inv @ m @ sm for m in small_m]

    # Wyrównanie jasności (na małych kolorowych podglądach)
    progress(0.92, "Wyrównywanie jasności")
    t = thumbs[ref].shape[1] / full_size[0]
    tm, tm_inv = np.diag([t, t, 1.0]), np.diag([1 / t, 1 / t, 1.0])
    gains = []
    for i in range(n):
        if i == ref:
            gains.append(np.ones(3, np.float32))
            continue
        mt = tm @ matrices[i] @ tm_inv
        img = cv2.warpAffine(thumbs[i], mt[:2], thumbs[i].shape[1::-1],
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)
        gains.append(_gain(thumbs[ref], img))

    result = Alignment(size=full_size, ref_index=ref, matrices=matrices, gains=gains,
                       warnings=warnings)
    result.crop = _common_crop(matrices, full_size)
    progress(1.0, "Dopasowanie zakończone")
    return result


def identity(paths: Sequence[str]) -> Alignment:
    """Brak dopasowania (np. gdy zdjęcia są już wyrównane)."""
    img, _ = imageio.load(paths[0])
    h, w = img.shape[:2]
    n = len(paths)
    return Alignment(size=(w, h), ref_index=n // 2, matrices=[np.eye(3) for _ in range(n)],
                     gains=[np.ones(3, np.float32) for _ in range(n)], crop=(0, 0, w, h))
