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
                  fine_weight: float = 0.8, base_sharpness: float = 8.0,
                  halo_radius: float = 40.0, halo_strength: float = 0.0) -> np.ndarray:
    """neighborhood - promień oceny ostrości (sigma w pikselach warstwy),
    consistency - rozmiar filtra spójności (nieparzysty, 1 = wyłączony),
    fine_weight - 0..1, jak bardzo o wyborze klatki na KAŻDEJ warstwie decyduje
                  ostrość drobnych szczegółów (0 = klasyczny PMax, 1 = tylko drobne szczegóły).
    base_sharpness - jak zdecydowanie warstwa bazowa (ogólna jasność i kolor) jest brana
                  z ostrej klatki (0 = średnia wszystkich klatek).
    halo_radius, halo_strength - ochrona przed echem z paralaksy: wyraźnie ostry obiekt
                  "rozlewa" przewagę swojej klatki na otoczenie o tym promieniu (piksele
                  pełnej rozdzielczości), więc rozmyte, przesunięte krawędzie tego samego
                  obiektu z innych klatek nie trafiają obok niego. 0 = wyłączone.

    Dlaczego to ważne: rozmyta krawędź z nieostrej klatki ma dużo energii na
    średnich i grubych warstwach piramidy. Klasyczny PMax wybiera ją tam, co
    daje miękkość, halo i echo. Drobne szczegóły (najwyższe częstotliwości)
    istnieją tylko w naprawdę ostrej klatce, więc to one powinny decydować."""
    progress = progress or (lambda f, m: None)
    if n > 255:
        raise ValueError("Maksymalnie 255 zdjęć w jednym stosie")

    best_score: list[np.ndarray] | None = None
    best_idx: list[np.ndarray] = []
    levels = 0
    norm_e: list[float] = []
    norm_f: list[float] = []
    top_scores: list[np.ndarray] = []

    # Przebieg 1: dla każdej warstwy ustalamy, z której klatki brać szczegóły
    for i in range(n):
        progress(0.5 * i / n, f"Analiza ostrości {i + 1}/{n}")
        img = load(i)
        if best_score is None:
            levels = _num_levels(*img.shape[:2])
        pyr = _laplacian_pyramid(img, levels)

        # mapa ostrości drobnych szczegółów (dwie najdrobniejsze warstwy)
        fine = np.abs(pyr[0]).sum(axis=2)
        if levels > 1:
            fine += cv2.pyrUp(np.abs(pyr[1]).sum(axis=2), dstsize=(fine.shape[1], fine.shape[0]))
        fine = cv2.GaussianBlur(fine, (0, 0), neighborhood)

        scores = []
        for k in range(levels):
            e = _energy(pyr[k], neighborhood)
            if k > 0:
                fine = cv2.pyrDown(fine, dstsize=(e.shape[1], e.shape[0]))
            if len(norm_e) <= k:  # normalizacja wg pierwszej klatki - te same stałe dla wszystkich
                norm_e.append(float(e.mean()) + 1e-9)
                norm_f.append(float(fine.mean()) + 1e-9)
            sc = (1.0 - fine_weight) * e / norm_e[k] + fine_weight * fine / norm_f[k]
            if halo_strength > 0 and halo_radius > 0:
                r = halo_radius / (2 ** k)
                if r >= 1.0:
                    sc = sc + halo_strength * cv2.GaussianBlur(sc, (0, 0), r)
            scores.append(sc)
        # ostrość na poziomie warstwy bazowej (do miękkiego wyboru bazy)
        top = pyr[-1]
        fine_top = cv2.pyrDown(fine, dstsize=(top.shape[1], top.shape[0]))
        if len(norm_f) == levels:
            norm_f.append(float(fine_top.mean()) + 1e-9)
        top_scores.append(fine_top / norm_f[levels])
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

    # Wagi warstwy bazowej: miękko preferujemy klatkę ostrą w danym miejscu.
    # (Zwykła średnia wnosiłaby rozlane światło z rozmytych klatek = "mgiełka".)
    ts = np.stack(top_scores)
    ts = ts / np.maximum(ts.max(axis=0, keepdims=True), 1e-9)
    base_w = ts ** base_sharpness
    base_w /= base_w.sum(axis=0, keepdims=True)

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
        result[-1] += pyr[-1] * base_w[i][:, :, None]

    progress(1.0, "Gotowe")
    return np.clip(_collapse(result), 0.0, 1.0)


DEPTH_WORK = 2000   # rozdzielczość mapy głębi (dłuższy bok)


def estimate_depth(n: int, load: FrameLoader, progress: ProgressFn | None = None,
                   smoothing: float = 5.0, conf_low: float = 1.5, conf_high: float = 4.0
                   ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Mapa głębi (numer klatki, ułamkowy) i pewność 0..1, w zmniejszonej rozdzielczości.

    Ostrość = energia Laplasjanu liczona w PEŁNEJ rozdzielczości i dopiero potem uśredniana
    do mapy roboczej. Pewność = jak wyraźnie najostrzejsza klatka odstaje od średniej ze
    POZOSTAŁYCH klatek (gładkie, wszędzie rozmyte tło -> stosunek ~1 -> brak pewności).
    Porównanie z pozostałymi, a nie ze wszystkimi, sprawia, że miara działa tak samo
    dla 2 i dla 100 klatek."""
    progress = progress or (lambda f, m: None)
    best = second_idx = idx = total = None
    full = (0, 0)
    for i in range(n):
        progress(i / n, f"Mapa głębi {i + 1}/{n}")
        gray = cv2.cvtColor(load(i), cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        full = (w, h)
        lap = cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 0.8), cv2.CV_32F, ksize=3)
        e = cv2.GaussianBlur(lap * lap, (0, 0), smoothing)
        q = DEPTH_WORK / max(h, w)
        sw, sh = max(1, round(w * q)), max(1, round(h * q))
        e = cv2.resize(e, (sw, sh), interpolation=cv2.INTER_AREA)
        if best is None:
            best, idx, total = e.copy(), np.zeros((sh, sw), np.float32), e.copy()
        else:
            m = e > best
            best[m] = e[m]
            idx[m] = i
            total += e
    others = (total - best) / max(1, n - 1)
    ratio = best / (others + 1e-12)
    conf = np.clip((ratio - conf_low) / (conf_high - conf_low), 0.0, 1.0).astype(np.float32)

    # odszumienie mapy: mała mediana (3x3) usuwa pojedyncze przeskoki, a cienkie
    # struktury (łodyżki, włoski) szersze niż ~3 px mapy zachowują swoją głębokość
    d8 = np.round(idx * 255.0 / max(1, n - 1)).astype(np.uint8)
    d = cv2.medianBlur(d8, 3).astype(np.float32) * max(1, n - 1) / 255.0

    # wypełnianie niepewnych miejsc głębokością z pewnego otoczenia (wieloskalowo)
    filled = d * conf
    weight = conf.copy()
    remaining = 1.0 - conf
    for sigma in (8, 32, 128):
        num = cv2.GaussianBlur(d * conf, (0, 0), sigma)
        den = cv2.GaussianBlur(conf, (0, 0), sigma)
        cand = num / np.maximum(den, 1e-6)
        take = remaining * np.clip(den * 4.0, 0.0, 1.0)
        filled += cand * take
        weight += take
        remaining = np.clip(remaining - take, 0.0, 1.0)
    fallback = float((d * conf).sum() / max(conf.sum(), 1e-6)) if conf.sum() > 0 else (n - 1) / 2
    depth = (filled + fallback * remaining) / np.maximum(weight + remaining, 1e-6)
    return depth.astype(np.float32), conf, full


def stack_depthmap(n: int, load: FrameLoader, progress: ProgressFn | None = None,
                   smoothing: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Odpowiednik DMAP. Zwraca (obraz, mapa głębi 0..1 w pełnej rozdzielczości).

    Każdy piksel pochodzi z klatki, w której jest ostry (bez wzmacniania szumu i kontrastu,
    wierne kolory); głębokość ułamkowa = płynne przejście między sąsiednimi klatkami."""
    progress = progress or (lambda f, m: None)
    depth_small, _, (w, h) = estimate_depth(n, load, lambda f, m: progress(0.5 * f, m),
                                            smoothing)
    depth = cv2.resize(depth_small, (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = np.floor(depth.min()), np.ceil(depth.max())
    acc = None
    wsum = np.zeros((h, w), np.float32)
    for i in range(n):
        progress(0.5 + 0.5 * i / n, f"Składanie {i + 1}/{n}")
        if i < lo - 1 or i > hi + 1:
            continue
        wgt = np.clip(1.0 - np.abs(depth - i), 0.0, 1.0)
        if not wgt.any():
            continue
        img = load(i)
        if acc is None:
            acc = np.zeros_like(img)
        acc += img * wgt[:, :, None]
        wsum += wgt
    progress(1.0, "Gotowe")
    out = acc / np.maximum(wsum, 1e-6)[:, :, None]
    return np.clip(out, 0.0, 1.0), depth / max(1, n - 1)
