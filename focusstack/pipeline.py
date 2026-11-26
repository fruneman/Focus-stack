"""Pełny proces: dopasowanie -> łączenie -> przycięcie. Używany przez GUI i testy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from . import align as align_mod
from . import imageio
from . import stack as stack_mod

ProgressFn = Callable[[float, str], None]


@dataclass
class Settings:
    method: str = "pyramid"        # "pyramid" (PMax+) lub "depthmap" (DMAP)
    do_align: bool = True
    crop: bool = True              # przytnij do obszaru wspólnego dla wszystkich klatek
    # pyramid
    neighborhood: float = 1.5
    consistency: int = 5
    fine_weight: float = 0.8
    base_sharpness: float = 8.0
    # depthmap
    smoothing: float = 6.0
    contrast: float = 6.0


class Cancelled(Exception):
    """Użytkownik przerwał obliczenia."""


@dataclass
class Result:
    image: np.ndarray
    bits: int
    alignment: align_mod.Alignment
    paths: list[str]
    crop: tuple[int, int, int, int] | None
    depth: np.ndarray | None = None

    def frame(self, index: int) -> np.ndarray:
        """Klatka źródłowa dopasowana i przycięta dokładnie jak wynik (do retuszu)."""
        img, _ = imageio.load(self.paths[index])
        img = self.alignment.warp(img, index)
        if self.crop:
            x, y, w, h = self.crop
            img = img[y:y + h, x:x + w]
        return np.ascontiguousarray(img)


def run(paths: Sequence[str], settings: Settings | None = None,
        progress: ProgressFn | None = None,
        cancelled: Callable[[], bool] | None = None) -> Result:
    settings = settings or Settings()
    user_progress = progress or (lambda f, m: None)

    def progress(f: float, m: str) -> None:
        if cancelled is not None and cancelled():
            raise Cancelled()
        user_progress(f, m)

    paths = list(paths)
    if len(paths) < 2:
        raise ValueError("Potrzebne są co najmniej 2 zdjęcia")

    def sub(a: float, b: float) -> ProgressFn:
        return lambda f, m: progress(a + (b - a) * f, m)

    if settings.do_align:
        al = align_mod.align(paths, progress=sub(0.0, 0.3))
    else:
        al = align_mod.identity(paths)

    bits = 8

    def load(i: int) -> np.ndarray:
        nonlocal bits
        img, b = imageio.load(paths[i])
        bits = max(bits, b)
        return al.warp(img, i)

    depth = None
    if settings.method == "depthmap":
        img, depth = stack_mod.stack_depthmap(len(paths), load, sub(0.3, 1.0),
                                              settings.smoothing, settings.contrast)
    else:
        img = stack_mod.stack_pyramid(len(paths), load, sub(0.3, 1.0), settings.neighborhood,
                                      settings.consistency, settings.fine_weight,
                                      settings.base_sharpness)

    crop = al.crop if settings.crop else None
    if crop:
        x, y, w, h = crop
        img = np.ascontiguousarray(img[y:y + h, x:x + w])
        if depth is not None:
            depth = depth[y:y + h, x:x + w]

    return Result(image=img, bits=bits, alignment=al, paths=paths, crop=crop, depth=depth)
