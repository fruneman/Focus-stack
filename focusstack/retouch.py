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

    def end_stroke(self) -> None:
        if self._stroke:
            self._undo.append(self._stroke)
            self._redo.clear()
        self._stroke = None

    def _save_tiles(self, y0: int, y1: int, x0: int, x1: int) -> None:
        store = self._stroke if self._stroke is not None else {}
        for ty in range(y0 // TILE, (y1 - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x1 - 1) // TILE + 1):
                if (ty, tx) not in store:
                    store[(ty, tx)] = self.image[ty * TILE:(ty + 1) * TILE,
                                                 tx * TILE:(tx + 1) * TILE].copy()

    def dab(self, index: int, x: float, y: float, radius: float,
            hardness: float = 0.5, opacity: float = 1.0) -> tuple[int, int, int, int] | None:
        """Jedno "dotknięcie" pędzla. Zwraca zmieniony prostokąt (x0, y0, x1, y1)."""
        h, w = self.image.shape[:2]
        r = max(1.0, radius)
        x0, x1 = max(0, int(x - r)), min(w, int(np.ceil(x + r)) + 1)
        y0, y1 = max(0, int(y - r)), min(h, int(np.ceil(y + r)) + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        src = self.frame(index)
        self._save_tiles(y0, y1, x0, x1)

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / r
        inner = np.clip(hardness, 0.0, 0.99)
        mask = np.clip((1.0 - d) / (1.0 - inner), 0.0, 1.0)
        mask = (mask * mask * (3 - 2 * mask)) * opacity  # gładkie wygaszanie krawędzi

        dst = self.image[y0:y1, x0:x1]
        m = mask[:, :, None]
        dst[:] = dst * (1 - m) + src[y0:y1, x0:x1] * m
        return x0, y0, x1, y1

    def line(self, index: int, xa: float, ya: float, xb: float, yb: float, radius: float,
             hardness: float = 0.5) -> tuple[int, int, int, int] | None:
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
