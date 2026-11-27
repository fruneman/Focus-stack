"""Dopasowanie (align) serii zdjęć do focus stackingu.

Strategia:
  * Klatką odniesienia jest środkowa klatka serii.
  * Każdą klatkę dopasowujemy do jej już dopasowanego sąsiada (sąsiednie
    klatki mają najbardziej podobną ostrość, więc dopasowanie jest pewne),
    a przekształcenia składamy w łańcuch aż do klatki odniesienia.
  * Pojedyncze dopasowanie: najpierw punkty charakterystyczne SIFT + RANSAC
    (zgrubnie, odporne na duże przesunięcia), potem dokładne doprecyzowanie
    algorytmem ECC (subpikselowo). Model ruchu: przesunięcie + obrót + skala
    (skala koryguje "oddychanie" obiektywu przy zmianie ostrości).
  * Na koniec wyrównanie jasności (osobno dla każdego kanału koloru).

Konwencja: macierz M (3x3) mapuje współrzędne klatki odniesienia na
współrzędne danej klatki. Warp wykonujemy z flagą WARP_INVERSE_MAP.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np

from . import imageio
from .errors import StackError

ProgressFn = Callable[[float, str], None]


@dataclass
class Alignment:
    size: tuple[int, int]                 # (szerokość, wysokość) pełnej rozdzielczości
    ref_index: int
    matrices: list[np.ndarray]            # 3x3, ref -> klatka, pełna rozdzielczość
    gains: list[np.ndarray] = field(default_factory=list)  # (3,) mnożnik BGR
    crop: tuple[int, int, int, int] | None = None          # x, y, w, h wspólnego obszaru
    warnings: list[str] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)       # klatki, których nie dopasowano
    # lokalne odkształcenie (korekta paralaksy): pole przesunięć w małej rozdzielczości,
    # w pikselach pełnej rozdzielczości, w układzie klatki odniesienia; None = brak
    flows: list[np.ndarray | None] = field(default_factory=list)

    def subset(self, keep: Sequence[int]) -> "Alignment":
        """Dopasowanie ograniczone do wybranych klatek (np. bez nieudanych)."""
        return Alignment(size=self.size, ref_index=list(keep).index(self.ref_index),
                         matrices=[self.matrices[i] for i in keep],
                         gains=[self.gains[i] for i in keep] if self.gains else [],
                         crop=self.crop, warnings=self.warnings, failed=[],
                         flows=[self.flows[i] for i in keep] if self.flows else [])

    def warp(self, img: np.ndarray, index: int) -> np.ndarray:
        w, h = self.size
        m = self.matrices[index][:2]
        flow = self.flows[index] if self.flows else None
        if flow is None:
            out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT)
        else:
            # jedno przepróbkowanie: punkt x układu odniesienia -> M·(x + f(x)) w klatce
            f = cv2.resize(flow.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            xx += f[..., 0]
            yy += f[..., 1]
            del f
            mx = (m[0, 0] * xx + m[0, 1] * yy + m[0, 2]).astype(np.float32)
            my = (m[1, 0] * xx + m[1, 1] * yy + m[1, 2]).astype(np.float32)
            del xx, yy
            out = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        if self.gains:
            out *= self.gains[index][None, None, :]
        return out


def _to_similarity(m: np.ndarray) -> np.ndarray:
    """Rzutuje macierz afiniczną 2x3 na podobieństwo (bez ścinania / niejednorodnej skali)."""
    a, b = m[0, 0], m[0, 1]
    c, d = m[1, 0], m[1, 1]
    s = 0.5 * (np.hypot(a, c) + np.hypot(b, d))
    ang = np.arctan2(c - b, a + d)
    out = m.copy()
    out[0, 0] = s * np.cos(ang)
    out[0, 1] = -s * np.sin(ang)
    out[1, 0] = s * np.sin(ang)
    out[1, 1] = s * np.cos(ang)
    return out


def _prep(gray: np.ndarray) -> np.ndarray:
    """Normalizacja kontrastu (CLAHE) - pomaga przy różnicach ostrości i jasności."""
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(u8)


class _Features:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=5000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.cache: dict[int, tuple] = {}

    def get(self, idx: int, img_u8: np.ndarray):
        if idx not in self.cache:
            self.cache[idx] = self.sift.detectAndCompute(img_u8, None)
        return self.cache[idx]


MIN_INLIERS = 20      # tyle zgodnych punktów SIFT uznajemy za wiarygodne dopasowanie
MIN_ECC_CC = 0.85     # minimalna korelacja ECC dla metody zapasowej


def _plausible(m: np.ndarray, shape: tuple[int, int]) -> bool:
    """Czy przekształcenie ma sens dla dwóch sąsiednich klatek serii makro?"""
    if m is None or not np.all(np.isfinite(m)):
        return False
    a, c = m[0, 0], m[1, 0]
    scale = float(np.hypot(a, c))
    rot = abs(float(np.degrees(np.arctan2(c, a))))
    h, w = shape
    center = m[:2, :2] @ np.array([w / 2, h / 2]) + m[:2, 2]
    shift = float(np.hypot(center[0] - w / 2, center[1] - h / 2))
    return 0.75 < scale < 1.33 and rot < 10.0 and shift < 0.45 * max(h, w)


def _sift_pair(feat: _Features, ia: int, a8: np.ndarray, ib: int, b8: np.ndarray):
    """SIFT + RANSAC. Zwraca (macierz 2x3, liczba zgodnych punktów) albo (None, n)."""
    ka, da = feat.get(ia, a8)
    kb, db = feat.get(ib, b8)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0
    matches = feat.matcher.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, len(good)
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    m, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                                         maxIters=5000, confidence=0.999)
    n_inl = int(inl.sum()) if inl is not None else 0
    if m is None or n_inl < MIN_INLIERS or not _plausible(m, a8.shape):
        return None, n_inl
    return m.astype(np.float32), n_inl


def _resize(g: np.ndarray, size: int) -> np.ndarray:
    h, w = g.shape
    s = size / max(h, w)
    return cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def _detail_mask(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> np.ndarray | None:
    """Maska (w układzie obrazu A) obszarów z wyraźnymi szczegółami w A lub w B."""
    h, w = a.shape
    bw = cv2.warpAffine(b, m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REFLECT)

    def detail(g):
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        return cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2)

    d = np.maximum(detail(a), detail(bw))
    thr = np.percentile(d, 92)
    mask = (d > thr).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    if mask.mean() < 0.02:  # za mało, by dopasowanie było wiarygodne
        return None
    return mask * 255


def _coarse_to_fine(a: np.ndarray, b: np.ndarray):
    """Metoda zapasowa dla klatek o bardzo różnej ostrości (SIFT nie znajduje wspólnych
    punktów): porównanie całych, rozmytych kształtów jasności. Najpierw zgrubne
    przesunięcie przez dopasowanie wzorca, potem ECC od małej do dużej rozdzielczości.
    Zwraca (macierz 2x3 A->B w rozdzielczości roboczej, korelacja) albo (None, 0)."""
    work = max(a.shape)
    a3 = cv2.GaussianBlur(_resize(a, 300), (0, 0), 3)
    b3 = cv2.GaussianBlur(_resize(b, 300), (0, 0), 3)
    h, w = b3.shape
    mg = 0.2
    tpl = b3[int(h * mg):int(h * (1 - mg)), int(w * mg):int(w * (1 - mg))]
    pad = cv2.copyMakeBorder(a3, h // 3, h // 3, w // 3, w // 3, cv2.BORDER_REPLICATE)
    res = cv2.matchTemplate(pad, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < 0.5:
        return None, 0.0
    tx = loc[0] - w // 3 - int(w * mg)
    ty = loc[1] - h // 3 - int(h * mg)
    m = np.float32([[1, 0, -tx], [0, 1, -ty]])

    cc, prev = 0.0, 300
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    for size, sig in ((300, 3.0), (600, 2.0), (1200, 1.2), (work, 1.0)):
        size = min(size, work)
        m[:, 2] *= size / prev
        prev = size
        af = cv2.GaussianBlur(_resize(a, size), (0, 0), sig).astype(np.float32)
        bf = cv2.GaussianBlur(_resize(b, size), (0, 0), sig).astype(np.float32)
        # Na dużych rozdzielczościach porównujemy tylko obszary z wyraźnymi szczegółami:
        # rozmyte tło przesuwa się inaczej niż ostry obiekt (paralaksa) i "ciągnęłoby"
        # dopasowanie w swoją stronę.
        mask = _detail_mask(af, bf, m) if size >= 1200 else None
        try:
            cc, m = cv2.findTransformECC(af, bf, m, cv2.MOTION_AFFINE, criteria, mask, 5)
        except cv2.error:
            if mask is None:
                return None, 0.0
            try:  # za mało szczegółów w masce - próbujemy na całym kadrze
                cc, m = cv2.findTransformECC(af, bf, m, cv2.MOTION_AFFINE, criteria, None, 5)
            except cv2.error:
                return None, 0.0
        if size == work:
            break
    m = _to_similarity(m)
    if not _plausible(m, a.shape) or cc < MIN_ECC_CC:
        return None, float(cc)
    return m.astype(np.float32), float(cc)


def _estimate_pair(feat: _Features, ia: int, a: np.ndarray, ib: int, b: np.ndarray
                   ) -> tuple[np.ndarray | None, str | None]:
    """Szacuje macierz 2x3 mapującą współrzędne obrazu A na obraz B (rozdzielczość robocza).
    Zwraca (macierz, ostrzeżenie) albo (None, powód), gdy dopasowanie się nie udało."""
    a8, b8 = _prep(a), _prep(b)

    # 1) SIFT + RANSAC, doprecyzowane przez ECC
    init, n_inl = _sift_pair(feat, ia, a8, ib, b8)
    if init is not None:
        af = cv2.GaussianBlur(a8.astype(np.float32) / 255.0, (0, 0), 1.0)
        bf = cv2.GaussianBlur(b8.astype(np.float32) / 255.0, (0, 0), 1.0)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        try:
            _, m = cv2.findTransformECC(af, bf, init.copy(), cv2.MOTION_AFFINE, criteria, None, 5)
            m = _to_similarity(m)
            drift = np.abs(m[:, 2] - init[:, 2]).max()
            scale_diff = abs(np.hypot(m[0, 0], m[1, 0]) - np.hypot(init[0, 0], init[1, 0]))
            if drift > 0.02 * max(a.shape) or scale_diff > 0.02 or not _plausible(m, a.shape):
                m = init
        except cv2.error:
            m = init
        return m.astype(np.float64), None

    # 2) Metoda zapasowa - porównanie całych kształtów
    m, cc = _coarse_to_fine(a, b)
    if m is not None:
        return m.astype(np.float64), "mało wspólnych szczegółów – dopasowano metodą zapasową"
    return None, ("nie pasuje do sąsiednich klatek (inne ujęcie albo zbyt duży skok "
                  "ostrości lub kadru)")


FLOW_SIZE = 1000      # rozdzielczość liczenia przepływu (dłuższy bok)


def _local_flow(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """Gładkie pole lokalnych przesunięć: ref(x) ≈ mov(x + f(x)), w pikselach tej skali.

    Przepływ optyczny (DIS) liczony na lekko rozmytych obrazach, uśredniany z wagą
    "ilości szczegółów" i mocno wygładzany - dopuszczamy tylko płynne odkształcenia
    (różne przesunięcie obiektów na różnej głębokości), nie drobne "falowanie".
    Tam, gdzie nie ma szczegółów (gładkie tło), odkształcenie wygasa do zera."""
    a = cv2.GaussianBlur(ref, (0, 0), 2.0)
    b = cv2.GaussianBlur(mov, (0, 0), 2.0)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f = dis.calc(np.clip(a * 255, 0, 255).astype(np.uint8),
                 np.clip(b * 255, 0, 255).astype(np.uint8), None)
    gm = np.maximum(a, b)
    gx = cv2.Sobel(gm, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gm, cv2.CV_32F, 0, 1)
    wgt = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2)
    smooth = 0.02 * max(ref.shape)
    wsum = cv2.GaussianBlur(wgt, (0, 0), smooth) + 1e-6
    out = np.dstack([cv2.GaussianBlur(f[..., k] * wgt, (0, 0), smooth) / wsum for k in range(2)])
    conf = np.clip(wsum / (2.0 * float(np.median(wsum)) + 1e-6), 0.0, 1.0)
    out *= conf[..., None]
    limit = 0.03 * max(ref.shape)
    mag = np.hypot(out[..., 0], out[..., 1])
    out *= np.minimum(1.0, limit / np.maximum(mag, 1e-6))[..., None]
    return out.astype(np.float32)


def _gain(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Mnożniki BGR wyrównujące średnią jasność do klatki odniesienia (środek kadru)."""
    h, w = ref.shape[:2]
    y0, y1, x0, x1 = h // 6, h - h // 6, w // 6, w - w // 6
    r = ref[y0:y1, x0:x1].reshape(-1, 3).mean(0)
    i = img[y0:y1, x0:x1].reshape(-1, 3).mean(0)
    g = r / np.maximum(i, 1e-6)
    return np.clip(g, 0.7, 1.4).astype(np.float32)


def _common_crop(matrices: Sequence[np.ndarray], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Prostokąt (w układzie odniesienia) pokryty przez wszystkie klatki."""
    w, h = size
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
    left, top, right, bottom = 0.0, 0.0, float(w), float(h)
    for m in matrices:
        if abs(np.linalg.det(m)) < 1e-6:
            continue
        p = np.linalg.inv(m) @ corners  # rogi klatki w układzie odniesienia
        left = max(left, p[0, 0], p[0, 3])
        right = min(right, p[0, 1], p[0, 2])
        top = max(top, p[1, 0], p[1, 1])
        bottom = min(bottom, p[1, 2], p[1, 3])
    x0, y0 = int(np.ceil(left)), int(np.ceil(top))
    x1, y1 = int(np.floor(right)), int(np.floor(bottom))
    if x1 - x0 < w // 2 or y1 - y0 < h // 2:  # coś poszło nie tak - nie przycinamy
        return 0, 0, w, h
    return x0, y0, x1 - x0, y1 - y0


def align(paths: Sequence[str], work_size: int = 2000,
          progress: ProgressFn | None = None, local: bool = False) -> Alignment:
    n = len(paths)
    if n == 0:
        raise StackError("Brak zdjęć", "Lista zdjęć jest pusta.", ["Dodaj zdjęcia serii."])
    progress = progress or (lambda f, m: None)

    smalls, thumbs = [], []
    full_size = None
    s = 1.0
    for i, p in enumerate(paths):
        progress(0.3 * i / n, f"Wczytywanie {i + 1}/{n}")
        img, _ = imageio.load(p)
        h, w = img.shape[:2]
        if full_size is None:
            full_size = (w, h)
            s = min(1.0, work_size / max(w, h))
        elif (w, h) != full_size:
            first = os.path.basename(paths[0])
            orient = ""
            if (w, h) == (full_size[1], full_size[0]):
                orient = (" Wygląda na to, że jedno zdjęcie jest pionowe, a drugie poziome "
                          "(aparat zapisał inną orientację).")
            raise StackError(
                "Zdjęcia mają różne wymiary",
                f"{os.path.basename(p)} ma {w}×{h} px, a {first} ma "
                f"{full_size[0]}×{full_size[1]} px. Wszystkie klatki serii muszą mieć ten sam "
                f"rozmiar.{orient}",
                ["Usuń z listy zdjęcia, które nie należą do serii (np. wykadrowane, "
                 "zmniejszone albo z innej sesji).",
                 "Jeśli różnią się tylko orientacją – obróć je w przeglądarce zdjęć tak samo "
                 "jak resztę serii."])
        small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        smalls.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        t = 400 / max(w, h)
        thumbs.append(cv2.resize(img, (round(w * t), round(h * t)), interpolation=cv2.INTER_AREA))
        del img

    ref = n // 2
    eye = np.eye(3)
    small_m: list[np.ndarray] = [eye.copy() for _ in range(n)]
    warnings: list[str] = []
    feat = _Features()

    failed: list[int] = []
    done = {ref}
    total = n - 1
    step = 0
    for direction in (1, -1):
        last = ref
        b = ref + direction
        while 0 <= b < n:
            step += 1
            progress(0.3 + 0.6 * step / max(1, total), f"Dopasowywanie {step}/{total}")
            # próbujemy z ostatnią dopasowaną klatką, a potem z 2 wcześniejszymi
            candidates = [last] + sorted((i for i in done if i != last),
                                         key=lambda i: abs(i - b))[:2]
            ok = False
            reason = None
            for a in candidates:
                m, msg = _estimate_pair(feat, a, smalls[a], b, smalls[b])
                if m is not None:
                    small_m[b] = np.vstack([m, [0, 0, 1]]) @ small_m[a]
                    if msg:
                        warnings.append(f"{os.path.basename(paths[b])}: {msg}")
                    done.add(b)
                    last = b
                    ok = True
                    break
                reason = reason or msg
            if not ok:
                failed.append(b)
                warnings.append(f"{os.path.basename(paths[b])}: pominięta – {reason}")
            b += direction

    ok_frames = n - len(failed)
    if ok_frames < 2 or len(failed) > n // 2:
        names = ", ".join(os.path.basename(paths[i]) for i in sorted(failed))
        raise StackError(
            "Nie udało się dopasować zdjęć",
            f"Program nie znalazł wspólnych elementów między klatkami: {names}. Zwykle znaczy "
            "to, że między kolejnymi zdjęciami ostrość albo kadr zmieniają się zbyt mocno, "
            "albo że na liście są zdjęcia z różnych ujęć.",
            ["Sprawdź, czy wszystkie zdjęcia na liście należą do tej samej serii.",
             "Dodaj brakujące klatki pośrednie – im mniejszy skok ostrości między kolejnymi "
             "zdjęciami, tym pewniejsze dopasowanie.",
             "Jeśli zdjęcia były robione ze statywu i się nie przesuwają – wyłącz "
             "„Dopasuj zdjęcia”."])

    # Przeliczenie na pełną rozdzielczość: M_full = S^-1 · M_small · S
    sm = np.diag([s, s, 1.0])
    sm_inv = np.diag([1 / s, 1 / s, 1.0])
    matrices = [sm_inv @ m @ sm for m in small_m]
    ok_idx = [i for i in range(n) if i not in failed]

    # Wyrównanie jasności (na małych kolorowych podglądach)
    progress(0.92, "Wyrównywanie jasności")
    t = thumbs[ref].shape[1] / full_size[0]
    tm, tm_inv = np.diag([t, t, 1.0]), np.diag([1 / t, 1 / t, 1.0])
    gains = []
    for i in range(n):
        if i == ref or i in failed:
            gains.append(np.ones(3, np.float32))
            continue
        mt = tm @ matrices[i] @ tm_inv
        img = cv2.warpAffine(thumbs[i], mt[:2], thumbs[i].shape[1::-1],
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)
        gains.append(_gain(thumbs[ref], img))

    flows: list[np.ndarray | None] = []
    if local:
        progress(0.95, "Korekta paralaksy")
        flows = _parallax_flows(smalls, small_m, ref, failed, full_size)

    result = Alignment(size=full_size, ref_index=ref, matrices=matrices, gains=gains,
                       warnings=warnings, failed=sorted(failed), flows=flows)
    result.crop = _common_crop([matrices[i] for i in ok_idx], full_size)
    progress(1.0, "Dopasowanie zakończone")
    return result


def _parallax_flows(smalls: list[np.ndarray], small_m: list[np.ndarray], ref: int,
                    failed: list[int], full_size: tuple[int, int]) -> list[np.ndarray | None]:
    """Lokalne odkształcenia klatek względem klatki odniesienia (łańcuchowo, od sąsiada
    do sąsiada - sąsiednie klatki mają podobną ostrość, więc przepływ jest pewniejszy)."""
    n = len(smalls)
    h, w = smalls[0].shape
    q = FLOW_SIZE / max(h, w)
    fw, fh = round(w * q), round(h * q)
    qm, qm_inv = np.diag([q, q, 1.0]), np.diag([1 / q, 1 / q, 1.0])

    def aligned(i):
        m = qm @ small_m[i] @ qm_inv
        g = cv2.resize(smalls[i], (fw, fh), interpolation=cv2.INTER_AREA)
        return cv2.warpAffine(g, m[:2], (fw, fh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT)

    xx, yy = np.meshgrid(np.arange(fw, dtype=np.float32), np.arange(fh, dtype=np.float32))
    to_full = full_size[0] / fw
    flows: list[np.ndarray | None] = [None] * n
    flows[ref] = np.zeros((fh, fw, 2), np.float32)
    ref_img = aligned(ref)
    for direction in (1, -1):
        prev_final = ref_img
        b = ref + direction
        while 0 <= b < n:
            if b not in failed:
                g = aligned(b)
                f = _local_flow(prev_final, g)      # w pikselach skali przepływu
                prev_final = cv2.remap(g, xx + f[..., 0], yy + f[..., 1], cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
                flows[b] = f * to_full
            b += direction
    for i in range(n):
        if flows[i] is not None:
            flows[i] = flows[i].astype(np.float16)  # oszczędność pamięci
    flows[ref] = None
    return flows


def identity(paths: Sequence[str]) -> Alignment:
    """Brak dopasowania (np. gdy zdjęcia są już wyrównane)."""
    img, _ = imageio.load(paths[0])
    h, w = img.shape[:2]
    n = len(paths)
    return Alignment(size=(w, h), ref_index=n // 2, matrices=[np.eye(3) for _ in range(n)],
                     gains=[np.ones(3, np.float32) for _ in range(n)], crop=(0, 0, w, h))
