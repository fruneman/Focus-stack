"""Podgląd zdjęcia: zoom kółkiem myszy, przesuwanie, pędzel retuszu."""

from __future__ import annotations

from typing import Callable

import threading

import cairo
import cv2
import numpy as np
from gi.repository import Gdk, GLib, Gtk

from ..retouch import Retoucher

MIN_MIP_SIDE = 400


def to_bgrx(img: np.ndarray) -> np.ndarray:
    """float32 BGR 0..1 (lub uint8 BGR) -> uint8 BGRX (format cairo RGB24)."""
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)


class _Mips:
    """Obraz w kilku rozdzielczościach (1, 1/2, 1/4, ...) jako powierzchnie cairo."""

    def __init__(self, img: np.ndarray):
        self.levels: list[np.ndarray] = [to_bgrx(img)]
        while max(self.levels[-1].shape[:2]) > MIN_MIP_SIDE:
            prev = self.levels[-1]
            h, w = prev.shape[:2]
            self.levels.append(cv2.resize(prev, ((w + 1) // 2, (h + 1) // 2),
                                          interpolation=cv2.INTER_AREA))
        self.surfaces = [self._surface(a) for a in self.levels]

    @staticmethod
    def _surface(arr: np.ndarray) -> cairo.ImageSurface:
        h, w = arr.shape[:2]
        return cairo.ImageSurface.create_for_data(memoryview(arr), cairo.FORMAT_RGB24, w, h,
                                                  arr.strides[0])

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.levels[0].shape[:2]
        return w, h

    def update(self, img: np.ndarray, box: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = box
        for surf in self.surfaces:
            surf.flush()  # wymagane przez cairo przed zmianą pikseli "pod spodem"
        base = self.levels[0]
        base[y0:y1, x0:x1] = to_bgrx(img[y0:y1, x0:x1])
        self.surfaces[0].mark_dirty_rectangle(x0, y0, x1 - x0, y1 - y0)
        for k in range(1, len(self.levels)):
            f = 2 ** k
            lv = self.levels[k]
            lh, lw = lv.shape[:2]
            a0, b0 = x0 // f, y0 // f
            a1, b1 = min(lw, -(-x1 // f)), min(lh, -(-y1 // f))
            src = base[b0 * f:min(base.shape[0], b1 * f), a0 * f:min(base.shape[1], a1 * f)]
            lv[b0:b1, a0:a1] = cv2.resize(src, (a1 - a0, b1 - b0), interpolation=cv2.INTER_AREA)
            self.surfaces[k].mark_dirty_rectangle(a0, b0, a1 - a0, b1 - b0)


class ImageViewer(Gtk.DrawingArea):
    def __init__(self):
        super().__init__(hexpand=True, vexpand=True, focusable=True)
        self._mips: _Mips | None = None
        self.scale = 1.0          # piksele ekranu na piksel obrazu
        self.ox = self.oy = 0.0   # współrzędne obrazu w lewym górnym rogu widoku
        self._fit = True
        self.paint_mode = False
        self.brush_radius = 40.0  # w pikselach obrazu
        self._cursor: tuple[float, float] | None = None
        # podgląd źródła pod pędzlem
        self.brush_preview = True
        self.brush_hardness = 0.3 * 0.95
        self._src: _Mips | None = None
        self._src_key = None
        self._src_gen = 0
        self._pan_start: tuple[float, float] | None = None
        self._painting = False
        self.on_paint: Callable[[str, float, float], None] | None = None  # "begin"/"move"/"end"
        self.on_zoom_changed: Callable[[float], None] | None = None
        self.on_brush_changed: Callable[[float], None] | None = None

        self.set_draw_func(self._draw)

        pan = Gtk.GestureDrag(button=0)
        pan.connect("drag-begin", self._drag_begin)
        pan.connect("drag-update", self._drag_update)
        pan.connect("drag-end", self._drag_end)
        self.add_controller(pan)
        self._drag = pan

        scroll = Gtk.EventControllerScroll(flags=Gtk.EventControllerScrollFlags.VERTICAL)
        scroll.connect("scroll", self._scroll)
        self.add_controller(scroll)

        motion = Gtk.EventControllerMotion()
        motion.connect("motion", self._motion)
        motion.connect("leave", lambda *_: self._set_cursor(None))
        self.add_controller(motion)

        self.connect("resize", lambda *_: self._fit and self.fit())

    # --- obraz --------------------------------------------------------------
    def set_image(self, img: np.ndarray | None, keep_view: bool = True) -> None:
        old = self._mips.size if self._mips else None
        self._mips = _Mips(img) if img is not None else None
        if not keep_view or old is None or (self._mips and self._mips.size != old):
            self.fit()
        self.queue_draw()

    def update_region(self, img: np.ndarray, box: tuple[int, int, int, int]) -> None:
        if self._mips:
            self._mips.update(img, box)
            self.queue_draw()

    def set_brush_source(self, key, img: np.ndarray | None) -> None:
        """Obraz pokazywany pod pędzlem (klatka lub wynik drugiej metody).
        Pomniejszenia budowane są w tle; `key` pozwala nie budować ich ponownie."""
        if img is None:
            self._src_gen += 1
            self._src, self._src_key = None, None
            self.queue_draw()
            return
        if key == self._src_key and self._src is not None:
            return
        self._src_gen += 1
        gen = self._src_gen
        self._src_key = key
        self._src = None

        def work():
            mips = _Mips(img)
            GLib.idle_add(self._src_ready, gen, mips)

        threading.Thread(target=work, daemon=True).start()

    def _src_ready(self, gen, mips):
        if gen == self._src_gen:
            self._src = mips
            self.queue_draw()
        return False

    @property
    def has_image(self) -> bool:
        return self._mips is not None

    # --- widok ---------------------------------------------------------------
    def fit(self) -> None:
        self._fit = True
        if not self._mips:
            return
        w, h = self._mips.size
        vw, vh = max(1, self.get_width()), max(1, self.get_height())
        self.scale = min(vw / w, vh / h) * 0.98
        self.ox = (w - vw / self.scale) / 2
        self.oy = (h - vh / self.scale) / 2
        self._zoom_changed()
        self.queue_draw()

    def zoom_to(self, scale: float, cx: float | None = None, cy: float | None = None) -> None:
        if not self._mips:
            return
        cx = self.get_width() / 2 if cx is None else cx
        cy = self.get_height() / 2 if cy is None else cy
        ix, iy = self.ox + cx / self.scale, self.oy + cy / self.scale
        self.scale = float(np.clip(scale, 0.02, 16.0))
        self.ox, self.oy = ix - cx / self.scale, iy - cy / self.scale
        self._fit = False
        self._zoom_changed()
        self.queue_draw()

    def _zoom_changed(self) -> None:
        if self.on_zoom_changed:
            self.on_zoom_changed(self.scale)

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        return self.ox + x / self.scale, self.oy + y / self.scale

    # --- rysowanie -------------------------------------------------------------
    def _paint_mips(self, cr: cairo.Context, mips: _Mips) -> None:
        dev = self.get_scale_factor()
        eff = self.scale * dev
        level = 0
        while level + 1 < len(mips.levels) and eff <= 0.5 ** (level + 1):
            level += 1
        s_l = 0.5 ** level
        cr.save()
        cr.translate(-self.ox * self.scale, -self.oy * self.scale)
        cr.scale(self.scale / s_l, self.scale / s_l)
        cr.set_source_surface(mips.surfaces[level], 0, 0)
        pattern = cr.get_source()
        pattern.set_filter(cairo.FILTER_NEAREST if eff / s_l >= 2.0 else cairo.FILTER_GOOD)
        w, h = mips.levels[level].shape[1], mips.levels[level].shape[0]
        cr.rectangle(0, 0, w, h)
        cr.fill()
        cr.restore()

    def _draw(self, area, cr: cairo.Context, width: int, height: int) -> None:
        if not self._mips:
            return
        self._paint_mips(cr, self._mips)

        if self.paint_mode and self._cursor:
            x, y = self._cursor
            r = self.brush_radius * self.scale
            if self.brush_preview and self._src is not None and self._src.size == self._mips.size:
                # źródło widoczne przez maskę o profilu pędzla = dokładnie to, co zostanie wklejone
                cr.save()
                cr.rectangle(x - r - 1, y - r - 1, 2 * r + 2, 2 * r + 2)
                cr.clip()
                cr.push_group()
                self._paint_mips(cr, self._src)
                cr.pop_group_to_source()
                grad = cairo.RadialGradient(x, y, 0, x, y, max(r, 1.0))
                for d in np.linspace(0.0, 1.0, 13):
                    a = float(Retoucher.brush_mask(np.array([d], np.float32), self.brush_hardness)[0])
                    grad.add_color_stop_rgba(float(d), 0, 0, 0, a)
                cr.mask(grad)
                cr.restore()
            cr.set_line_width(1.5)
            cr.arc(x, y, r, 0, 2 * np.pi)
            cr.set_source_rgba(0, 0, 0, 0.8)
            cr.stroke_preserve()
            cr.set_dash([4, 4])
            cr.set_source_rgba(1, 1, 1, 0.9)
            cr.stroke()

    # --- mysz ---------------------------------------------------------------------
    def _set_cursor(self, pos) -> None:
        self._cursor = pos
        self.set_cursor(Gdk.Cursor.new_from_name("none" if (pos and self.paint_mode) else "default"))
        if self.paint_mode:
            self.queue_draw()

    def _motion(self, ctrl, x, y) -> None:
        self._set_cursor((x, y))

    def _drag_begin(self, gesture, x, y) -> None:
        self.grab_focus()
        button = gesture.get_current_button()
        if button == 1 and self.paint_mode and self._mips:
            self._painting = True
            if self.on_paint:
                self.on_paint("begin", *self.to_image(x, y))
        else:
            self._pan_start = (self.ox, self.oy)
            self.set_cursor(Gdk.Cursor.new_from_name("grabbing"))

    def _drag_update(self, gesture, dx, dy) -> None:
        ok, sx, sy = gesture.get_start_point()
        if self._painting:
            self._cursor = (sx + dx, sy + dy)
            if self.on_paint:
                self.on_paint("move", *self.to_image(sx + dx, sy + dy))
        elif self._pan_start:
            self.ox = self._pan_start[0] - dx / self.scale
            self.oy = self._pan_start[1] - dy / self.scale
            self._fit = False
            self.queue_draw()

    def _drag_end(self, gesture, dx, dy) -> None:
        if self._painting:
            self._painting = False
            if self.on_paint:
                self.on_paint("end", 0, 0)
        self._pan_start = None
        self._set_cursor(self._cursor)

    def _scroll(self, ctrl, dx, dy) -> bool:
        state = ctrl.get_current_event_state()
        if self.paint_mode and state & Gdk.ModifierType.CONTROL_MASK:
            self.brush_radius = float(np.clip(self.brush_radius * (1 / 1.1 if dy > 0 else 1.1), 2, 600))
            if self.on_brush_changed:
                self.on_brush_changed(self.brush_radius)
            self.queue_draw()
            return True
        cx, cy = self._cursor if self._cursor else (None, None)
        self.zoom_to(self.scale * (1 / 1.2 if dy > 0 else 1.2), cx, cy)
        return True
