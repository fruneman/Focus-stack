"""Test retuszu: pędzel kopiuje piksele z klatki źródłowej, cofanie przywraca stan."""
import numpy as np

from focusstack.retouch import Retoucher


def main() -> int:
    rng = np.random.default_rng(0)
    result = rng.random((600, 800, 3)).astype(np.float32)
    frames = {1: np.zeros_like(result)}
    before = result.copy()
    r = Retoucher(result, lambda i: frames[i])

    r.begin_stroke()
    r.line(1, 100, 100, 500, 300, radius=30, hardness=0.8)
    r.end_stroke()
    assert np.allclose(result[100, 100], 0), "środek pędzla powinien mieć piksele źródła"
    assert np.allclose(result[300, 100], before[300, 100]), "poza pędzlem bez zmian"

    r.undo()
    assert np.array_equal(result, before), "cofnięcie powinno przywrócić oryginał"
    r.redo()
    assert np.allclose(result[100, 100], 0), "ponowienie powinno przywrócić retusz"
    print("Retusz OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
