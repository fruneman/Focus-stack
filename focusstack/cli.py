"""Proste uruchamianie z terminala - do testów i automatyzacji (GUI jest głównym interfejsem).

python -m focusstack.cli wynik.jpg zdj1.jpg zdj2.jpg ... [--method depthmap]
"""

import argparse
import sys
import time

from . import imageio, pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Focus stacking")
    ap.add_argument("output")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--method", choices=["pyramid", "depthmap"], default="pyramid")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--consistency", type=int, default=5)
    ap.add_argument("--fine-weight", type=float, default=1.0)
    a = ap.parse_args()

    s = pipeline.Settings(method=a.method, do_align=not a.no_align, crop=not a.no_crop,
                          consistency=a.consistency, fine_weight=a.fine_weight)
    t0 = time.time()
    last = [-1]

    def prog(f, msg):
        p = int(f * 100)
        if p != last[0]:
            last[0] = p
            print(f"\r{p:3d}%  {msg:<40}", end="", file=sys.stderr, flush=True)

    res = pipeline.run(sorted(a.inputs), s, prog)
    print(file=sys.stderr)
    for w in res.alignment.warnings:
        print("Uwaga:", w, file=sys.stderr)
    imageio.save(a.output, res.image, res.bits, exif_source=sorted(a.inputs)[0])
    print(f"Zapisano {a.output} w {time.time() - t0:.1f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
