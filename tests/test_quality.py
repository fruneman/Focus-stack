"""Test jakości na syntetycznej serii: dokładność dopasowania + jakość stacku + echo.

Uruchomienie: python -m tests.test_quality
"""

from __future__ import annotations

import glob
import os
import sys
import time

import cv2
import numpy as np

from focusstack import imageio, pipeline
from tests.synth import make_series

OUT = os.path.join(os.path.dirname(__file__), "..", "test_output")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def main() -> int:
    frames_dir = os.path.join(OUT, "synth")
    for f in glob.glob(os.path.join(frames_dir, "*.jpg")):
        os.remove(f)
    info = make_series(frames_dir)
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    truth, alpha = info["truth"], info["alpha"]

    # strefa wokół czułek (tam powstaje echo), ale bez samych czułek
    near = cv2.dilate((alpha > 0.05).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    halo_zone = near & (alpha < 0.01)

    ok = True
    for method in ("pyramid", "depthmap"):
        t0 = time.time()
        res = pipeline.run(paths, pipeline.Settings(method=method, crop=False))
        dt = time.time() - t0

        # błąd dopasowania: porównanie z prawdziwymi macierzami (w pikselach, na rogach kadru)
        h, w = truth.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64).T
        err = 0.0
        for m_true, m_est in zip(info["matrices"], res.alignment.matrices):
            p_true = m_true.astype(np.float64) @ corners
            p_est = m_est[:2] @ corners
            err = max(err, float(np.abs(p_true - p_est).max()))

        b = 40  # pomijamy brzegi (odbicia na krawędziach)
        img = res.image[b:-b, b:-b]
        tr = truth[b:-b, b:-b]
        hz = halo_zone[b:-b, b:-b]
        p_all = psnr(img, tr)
        p_halo = psnr(img[hz], tr[hz])
        print(f"[{method:9s}] czas {dt:5.1f} s | maks. błąd dopasowania {err:5.2f} px | "
              f"PSNR całość {p_all:5.2f} dB | PSNR wokół czułek {p_halo:5.2f} dB")
        imageio.save(os.path.join(OUT, f"wynik_{method}.jpg"), res.image)
        if err > 1.0:
            ok = False

    # punkty odniesienia: najlepsza pojedyncza klatka i zwykła średnia
    best = max(psnr(imageio.load(p)[0][40:-40, 40:-40], truth[40:-40, 40:-40]) for p in paths)
    print(f"[odniesienie] najlepsza pojedyncza klatka (bez dopasowania): PSNR {best:5.2f} dB")
    imageio.save(os.path.join(OUT, "prawda.jpg"), truth)
    print("Dopasowanie OK" if ok else "BŁĄD: dopasowanie niedokładne")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
