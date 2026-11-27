"""Retusz wyniku pędzlem: przenoszenie fragmentów z wybranej (dopasowanej) klatki.

Typowe użycie - usuwanie echa/halo: w miejscu z "duchem" zamalowujemy wynik
fragmentem klatki, w której dany obiekt jest ostry i nie ma ducha.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np

TILE = 256


class Retoucher:
    def __init__(self, result: np.ndarray, frame_loader: Callable[[int], np.ndarray],
                 cache_size: int = 2):
        """result - wynik stackowania (float32 BGR 0..1), modyfikowany w miejscu.
        frame_loader(i) - zwraca klatkę i dopasowaną i przyciętą jak wynik."""
        self.image = result
        self._loader = frame_loader
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._undo: list[dict[tuple[int, int], np.ndarray]] = []
        self._redo: list[dict[tuple[int, int], np.ndarray]] = []
        self._stroke: dict[tuple[int, int], np.ndarray] | None = None
        self._mask: dict[tuple[int, int], np.ndarray] = {}  # krycie w bieżącym pociągnięciu

    # --- klatki źródłowe -------------------------------------------------
    def frame(self, index: int) -> np.ndarray:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        img = self._loader(index)
        if img.shape != self.image.shape:
            raise ValueError("Klatka źródłowa ma inny rozmiar niż wynik")
        self._cache[index] = img
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return img

    def has_frame(self, index: int) -> bool:
        return index in self._cache

    # --- pociągnięcia pędzla ----------------------------------------------
    def begin_stroke(self) -> None:
        self._stroke = {}
        self._mask = {}

    def end_stroke(self) -> None:
        if self._stroke:
            self._undo.append(self._stroke)
            self._redo.clear()
        self._stroke = None
        self._mask = {}

    @staticmethod
    def brush_mask(d: np.ndarray, hardness: float) -> np.ndarray:
        """Krycie pędzla w funkcji odległości od środka (d: 0 = środek, 1 = krawędź).

        hardness = część promienia kryjąca w 100%; dalej krycie gładko spada do zera dokładnie
        na brzegu kółka. Szerokość wtapiania zmienia się liniowo z twardością (0 -> wtapianie
        od środka do brzegu, ~1 -> ostra krawędź), a kółko kursora zawsze pokazuje zasięg."""
        h = float(np.clip(hardness, 0.0, 0.95))
        t = np.clip((1.0 - d) / (1.0 - h), 0.0, 1.0)
        return t * t * (3 - 2 * t)                 # gładkie wygaszanie (smoothstep)

    def dab(self, index: int, x: float, y: float, radius: float,
            hardness: float = 0.3, opacity: float = 1.0) -> tuple[int, int, int, int] | None:
        """Jedno "dotknięcie" pędzla. Zwraca zmieniony prostokąt (x0, y0, x1, y1).

        W obrębie jednego pociągnięcia krycie każdego piksela to MAKSIMUM z kolejnych
        dotknięć (a nie ich suma), a mieszanie liczone jest zawsze od obrazu sprzed
        pociągnięcia. Dzięki temu nakładające się dotknięcia nie "utwardzają" krawędzi."""
        h, w = self.image.shape[:2]
        r = max(1.0, radius)
        x0, x1 = max(0, int(x - r)), min(w, int(np.ceil(x + r)) + 1)
        y0, y1 = max(0, int(y - r)), min(h, int(np.ceil(y + r)) + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        src = self.frame(index)
        single = self._stroke is None
        if single:
            self.begin_stroke()

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / r
        mask = self.brush_mask(d, hardness) * opacity

        for ty in range(y0 // TILE, (y1 - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x1 - 1) // TILE + 1):
                ys, xs = ty * TILE, tx * TILE
                key = (ty, tx)
                if key not in self._stroke:
                    self._stroke[key] = self.image[ys:ys + TILE, xs:xs + TILE].copy()
                orig = self._stroke[key]
                if key not in self._mask:
                    self._mask[key] = np.zeros(orig.shape[:2], np.float32)
                tm = self._mask[key]
                # część wspólna kafelka i dotknięcia
                a0, a1 = max(x0, xs), min(x1, xs + orig.shape[1])
                b0, b1 = max(y0, ys), min(y1, ys + orig.shape[0])
                lm = tm[b0 - ys:b1 - ys, a0 - xs:a1 - xs]
                np.maximum(lm, mask[b0 - y0:b1 - y0, a0 - x0:a1 - x0], out=lm)
                m = lm[:, :, None]
                o = orig[b0 - ys:b1 - ys, a0 - xs:a1 - xs]
                self.image[b0:b1, a0:a1] = o * (1 - m) + src[b0:b1, a0:a1] * m

        if single:
            self.end_stroke()
        return x0, y0, x1, y1

    def line(self, index: int, xa: float, ya: float, xb: float, yb: float, radius: float,
             hardness: float = 0.3) -> tuple[int, int, int, int] | None:
        """Ciągła linia "dotknięć" (odstęp 1/4 promienia) - płynne pociągnięcie."""
        dist = float(np.hypot(xb - xa, yb - ya))
        steps = max(1, int(dist / max(1.0, radius * 0.25)))
        box = None
        for s in range(1, steps + 1):
            t = s / steps
            b = self.dab(index, xa + (xb - xa) * t, ya + (yb - ya) * t, radius, hardness)
            if b:
                box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                             max(box[2], b[2]), max(box[3], b[3]))
        return box

    # --- cofanie -------------------------------------------------------------
    def _swap(self, src: list, dst: list) -> list[tuple[int, int, int, int]]:
        if not src:
            return []
        tiles = src.pop()
        inverse = {}
        boxes = []
        for (ty, tx), patch in tiles.items():
            ys, xs = ty * TILE, tx * TILE
            region = self.image[ys:ys + patch.shape[0], xs:xs + patch.shape[1]]
            inverse[(ty, tx)] = region.copy()
            region[:] = patch
            boxes.append((xs, ys, xs + patch.shape[1], ys + patch.shape[0]))
        dst.append(inverse)
        return boxes

    def undo(self) -> list[tuple[int, int, int, int]]:
        return self._swap(self._undo, self._redo)

    def redo(self) -> list[tuple[int, int, int, int]]:
        return self._swap(self._redo, self._undo)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)
