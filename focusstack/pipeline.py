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
    fine_weight: float = 1.0
    # depthmap
    smoothing: float = 4.0
    blend: float = 2.0


@dataclass
class Result:
    image: np.ndarray
    bits: int
    alignment: align_mod.Alignment
    depth: np.ndarray | None = None


def run(paths: Sequence[str], settings: Settings | None = None,
        progress: ProgressFn | None = None) -> Result:
    settings = settings or Settings()
    progress = progress or (lambda f, m: None)
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
                                              settings.smoothing, settings.blend)
    else:
        img = stack_mod.stack_pyramid(len(paths), load, sub(0.3, 1.0), settings.neighborhood,
                                      settings.consistency, settings.fine_weight)

    if settings.crop and al.crop:
        x, y, w, h = al.crop
        img = img[y:y + h, x:x + w]
        if depth is not None:
            depth = depth[y:y + h, x:x + w]

    return Result(image=img, bits=bits, alignment=al, depth=depth)
