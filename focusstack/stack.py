"""Łączenie (stacking) dopasowanych klatek.

Dwie metody:

1. "pyramid" (odpowiednik PMax, z redukcją echa)
   Każda klatka jest rozkładana na piramidę Laplace'a (warstwy szczegółów od
   najdrobniejszych do najgrubszych). Na każdej warstwie wybieramy klatkę,
   która ma tam najwięcej szczegółów. Przeciw echu / halo:
     * ostrość oceniana w otoczeniu piksela, a nie w pojedynczym pikselu,
     * decyzja na grubszej warstwie uwzględnia ostrość z drobniejszych warstw
       (rozmyta, ale kontrastowa krawędź z nieostrej klatki ma dużo energii
       tylko na grubych warstwach, więc nie "wygrywa" z ostrą klatką),
     * filtr spójności (mediana) na mapie wyboru klatek usuwa odstające wybory.

2. "depthmap" (odpowiednik DMAP)
   Dla każdego piksela wybieramy klatkę o największej ostrości, mapę wyboru
   mocno wygładzamy i składamy obraz z płynnymi przejściami. Daje mniej
   artefaktów na gładkich tłach, ale gorzej radzi sobie z nakładającymi się
   strukturami (włoski, czułki na tle innych elementów).

Obie metody działają strumieniowo (dwa przebiegi przez klatki), więc nie
trzymają całej serii w pamięci.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

ProgressFn = Callable[[float, str], None]
FrameLoader = Callable[[int], np.ndarray]  # indeks -> dopasowana klatka float32 BGR


def _num_levels(h: int, w: int) -> int:
    return int(max(1, min(8, np.floor(np.log2(min(h, w) / 16.0)))))


def _laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = []
    cur = img
    for _ in range(levels):
        down = cv2.pyrDown(cur)
        up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
        pyr.append(cur - up)
        cur = down
    pyr.append(cur)  # najgrubsza warstwa (rozmyty obraz bazowy)
    return pyr


def _collapse(pyr: list[np.ndarray]) -> np.ndarray:
    cur = pyr[-1]
    for lap in reversed(pyr[:-1]):
        cur = cv2.pyrUp(cur, dstsize=(lap.shape[1], lap.shape[0])) + lap
    return cur


def _energy(lap: np.ndarray, sigma: float) -> np.ndarray:
    e = np.abs(lap).sum(axis=2)
    return cv2.GaussianBlur(e, (0, 0), sigma)


def _median(idx: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return idx
    if idx.dtype != np.uint8:  # medianBlur dla k>5 wymaga uint8
        return idx
    return cv2.medianBlur(idx, k)


def stack_pyramid(n: int, load: FrameLoader, progress: ProgressFn | None = None,
                  neighborhood: float = 1.5, consistency: int = 5,
                  fine_weight: float = 1.0) -> np.ndarray:
    """neighborhood - promień oceny ostrości (sigma w pikselach warstwy),
    consistency - rozmiar filtra spójności (nieparzysty, 1 = wyłączony),
    fine_weight - jak mocno decyzje drobnych warstw wpływają na grubsze (0 = klasyczny PMax)."""
    progress = progress or (lambda f, m: None)
    if n > 255:
        raise ValueError("Maksymalnie 255 zdjęć w jednym stosie")

    best_score: list[np.ndarray] | None = None
    best_idx: list[np.ndarray] = []
    levels = 0

    # Przebieg 1: dla każdej warstwy ustalamy, z której klatki brać szczegóły
    for i in range(n):
        progress(0.5 * i / n, f"Analiza ostrości {i + 1}/{n}")
        img = load(i)
        if best_score is None:
            levels = _num_levels(*img.shape[:2])
        pyr = _laplacian_pyramid(img, levels)
        scores = []
        prev = None
        for k in range(levels):
            e = _energy(pyr[k], neighborhood)
            if prev is not None and fine_weight > 0:
                e = e + fine_weight * cv2.pyrDown(prev, dstsize=(e.shape[1], e.shape[0]))
            scores.append(e)
            prev = e
        if best_score is None:
            best_score = scores
            best_idx = [np.zeros(s.shape, np.uint8) for s in scores]
        else:
            for k in range(levels):
                mask = scores[k] > best_score[k]
                best_score[k][mask] = scores[k][mask]
                best_idx[k][mask] = i

    # Filtr spójności
    best_idx = [_median(idx, consistency) for idx in best_idx]
    del best_score

    # Przebieg 2: składanie piramidy wynikowej
    result: list[np.ndarray] | None = None
    for i in range(n):
        progress(0.5 + 0.5 * i / n, f"Składanie {i + 1}/{n}")
        pyr = _laplacian_pyramid(load(i), levels)
        if result is None:
            result = [np.zeros_like(p) for p in pyr]
        for k in range(levels):
            m = best_idx[k] == i
            result[k][m] = pyr[k][m]
        result[-1] += pyr[-1] / n  # baza: średnia (stabilne kolory i jasność)

    progress(1.0, "Gotowe")
    return np.clip(_collapse(result), 0.0, 1.0)


def stack_depthmap(n: int, load: FrameLoader, progress: ProgressFn | None = None,
                   smoothing: float = 4.0, blend: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Zwraca (obraz, mapa głębi 0..1). smoothing - wygładzenie mapy ostrości,
    blend - szerokość płynnego przejścia między klatkami (sigma w pikselach)."""
    progress = progress or (lambda f, m: None)

    best = None
    idx = None
    for i in range(n):
        progress(0.5 * i / n, f"Analiza ostrości {i + 1}/{n}")
        gray = cv2.cvtColor(load(i), cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 0.8), cv2.CV_32F, ksize=3)
        s = cv2.GaussianBlur(lap * lap, (0, 0), smoothing)
        if best is None:
            best, idx = s, np.zeros(s.shape, np.uint8 if n <= 255 else np.uint16)
        else:
            m = s > best
            best[m] = s[m]
            idx[m] = i

    # Wygładzenie mapy wyboru: mediana na zmniejszonej mapie (szybko i skutecznie)
    h, w = idx.shape
    f = 4
    small = cv2.resize(idx, (w // f, h // f), interpolation=cv2.INTER_NEAREST)
    if small.dtype == np.uint8:
        small = cv2.medianBlur(small, 5)
    idx = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    del best

    acc = None
    wsum = np.zeros((h, w), np.float32)
    for i in range(n):
        progress(0.5 + 0.5 * i / n, f"Składanie {i + 1}/{n}")
        m = (idx == i).astype(np.float32)
        if not m.any():
            continue
        wgt = cv2.GaussianBlur(m, (0, 0), blend) if blend > 0 else m
        img = load(i)
        if acc is None:
            acc = np.zeros_like(img)
        acc += img * wgt[:, :, None]
        wsum += wgt

    progress(1.0, "Gotowe")
    out = acc / np.maximum(wsum, 1e-6)[:, :, None]
    depth = idx.astype(np.float32) / max(1, n - 1)
    return np.clip(out, 0.0, 1.0), depth
