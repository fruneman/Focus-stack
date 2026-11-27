"""Regresja: seria z dużymi skokami ostrości i kadru (wcześniej "Singular matrix").

Każda klatka musi dostać sensowne przekształcenie (bez skali 0) albo zostać
jawnie pominięta - nigdy zdegenerowana macierz.
"""
import glob
import os
import sys

import numpy as np

from focusstack import align

SERIES = os.path.join(os.path.dirname(__file__), "..", "Seria-z-bledem")


def main() -> int:
    paths = sorted(glob.glob(os.path.join(SERIES, "*.jpg")))
    if not paths:
        print("Brak serii testowej - pomijam (przywrócenie: zob. README, sekcja Testy)")
        return 0
    al = align.align(paths)
    w, h = al.size
    for i, m in enumerate(al.matrices):
        if i in al.failed:
            continue
        scale = float(np.hypot(m[0, 0], m[1, 0]))
        assert 0.75 < scale < 1.33, f"{os.path.basename(paths[i])}: skala {scale:.3f}"
        assert abs(np.linalg.det(m)) > 0.5, f"{os.path.basename(paths[i])}: macierz zdegenerowana"
    assert al.crop[2] < w or al.crop[3] < h, "przycięcie powinno uwzględnić przesunięcia"
    print(f"Dopasowanie odporne OK (pominięte: {len(al.failed)}, ostrzeżenia: {len(al.warnings)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
