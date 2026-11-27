"""Główne okno aplikacji."""

from __future__ import annotations

import os
import threading
import traceback

import numpy as np
from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk
from PIL import Image

from .. import imageio, pipeline
from ..errors import StackError
from ..retouch import Retoucher
from .viewer import ImageViewer

THUMB = 72
METHODS = [("pyramid", "Piramida"),
           ("depthmap", "Mapa głębi")]


def _thumbnail(path: str) -> tuple[bytes, int, int]:
    with Image.open(path) as im:
        im.draft("RGB", (THUMB * 4, THUMB * 4))  # szybkie dekodowanie JPG w niższej rozdzielczości
        im = im.convert("RGB")
        try:
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im)
        except Exception:
            pass
        im.thumbnail((THUMB * 2, THUMB * 2))
        return im.tobytes(), im.width, im.height


class FrameRow(Adw.ActionRow):
    def __init__(self, path: str, on_remove):
        super().__init__(title=GLib.markup_escape_text(os.path.basename(path)), activatable=True)
        self.path = path
        self.picture = Gtk.Picture(content_fit=Gtk.ContentFit.COVER, can_shrink=True)
        self.picture.set_size_request(THUMB, THUMB * 2 // 3)
        frame = Gtk.Frame(child=self.picture, valign=Gtk.Align.CENTER)
        frame.add_css_class("thumb")
        self.add_prefix(frame)
        rm = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER,
                        tooltip_text="Usuń z listy")
        rm.add_css_class("flat")
        rm.connect("clicked", lambda *_: on_remove(self))
        self.add_suffix(rm)
        self.badge = Gtk.Image(icon_name="document-edit-symbolic", visible=False,
                               tooltip_text="Źródło retuszu")
        self.add_suffix(self.badge)

    def set_thumb(self, data: bytes, w: int, h: int) -> None:
        tex = Gdk.MemoryTexture.new(w, h, Gdk.MemoryFormat.R8G8B8, GLib.Bytes.new(data), w * 3)
        self.picture.set_paintable(tex)


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application):
        super().__init__(application=app, title="Focus Stack", default_width=1280,
                         default_height=820)
        self.paths: list[str] = []
        self.rows: list[FrameRow] = []
        self.selected: int | None = None
        self.result: pipeline.Result | None = None
        self.retoucher: Retoucher | None = None
        self.settings = pipeline.Settings()
        self.dirty = False
        self._cancel = False
        self._busy = False
        self._last_paint: tuple[float, float] | None = None
        self._showing = "result"
        self._lock = threading.Lock()

        self._build_ui()
        self._setup_actions()
        self.connect("close-request", self._on_close_request)
        self._update_state()

    # =====================================================================
    # Budowa interfejsu
    # =====================================================================
    def _build_ui(self) -> None:
        self.toasts = Adw.ToastOverlay()
        self.set_content(self.toasts)

        split = Adw.OverlaySplitView(sidebar_width_fraction=0.26, min_sidebar_width=300,
                                     max_sidebar_width=400)
        self.split = split
        self.toasts.set_child(split)

        # --- pasek boczny ---
        side_view = Adw.ToolbarView()
        side_header = Adw.HeaderBar(show_title=False)
        add_btn = Gtk.Button(icon_name="list-add-symbolic", tooltip_text="Dodaj zdjęcia (Ctrl+O)",
                             action_name="win.open")
        side_header.pack_start(add_btn)
        side_header.set_title_widget(Adw.WindowTitle(title="Zdjęcia"))
        side_view.add_top_bar(side_header)

        side = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18, margin_top=12,
                       margin_bottom=18, margin_start=12, margin_end=12)
        scroll = Gtk.ScrolledWindow(child=side, hscrollbar_policy=Gtk.PolicyType.NEVER,
                                    vexpand=True)
        side_view.set_content(scroll)
        split.set_sidebar(side_view)

        # lista zdjęć
        self.frames_group = Adw.PreferencesGroup(title="Klatki",
                                                 description="Kliknij klatkę, aby ją obejrzeć")
        self.listbox = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self.listbox.add_css_class("boxed-list")
        self.listbox.connect("row-activated", self._on_row_activated)
        self.frames_group.add(self.listbox)
        clear = Gtk.Button(icon_name="edit-clear-all-symbolic", tooltip_text="Wyczyść listę",
                           action_name="win.clear", valign=Gtk.Align.CENTER)
        clear.add_css_class("flat")
        self.frames_group.set_header_suffix(clear)
        side.append(self.frames_group)

        # ustawienia
        sg = Adw.PreferencesGroup(title="Ustawienia")
        self.method_row = Adw.ComboRow(title="Metoda",
                                       model=Gtk.StringList.new([m[1] for m in METHODS]))
        self.method_row.set_subtitle("Piramida - zalecana dla włosków i drobnych struktur")
        self.method_row.connect("notify::selected", self._on_method_changed)
        sg.add(self.method_row)
        self.align_row = Adw.SwitchRow(title="Dopasuj zdjęcia", active=True,
                                       subtitle="Wyłącz tylko dla zdjęć ze statywu i szyny")
        sg.add(self.align_row)
        self.local_row = Adw.SwitchRow(
            title="Korekta paralaksy", active=True,
            subtitle="Zgrywa obiekty na różnej głębokości – mniej ech przy zdjęciach z ręki")
        self.align_row.bind_property("active", self.local_row, "sensitive",
                                     GObject.BindingFlags.SYNC_CREATE)
        sg.add(self.local_row)
        self.crop_row = Adw.SwitchRow(title="Przytnij brzegi", active=True,
                                      subtitle="Usuwa krawędzie, których nie pokrywają wszystkie klatki")
        sg.add(self.crop_row)
        side.append(sg)

        # retusz
        self.retouch_group = Adw.PreferencesGroup(
            title="Retusz",
            description="Wybierz na liście klatkę, w której dany fragment jest ostry i bez "
                        "echa, a potem zamaluj go na wyniku.")
        self.retouch_row = Adw.SwitchRow(title="Pędzel retuszu",
                                         subtitle="Skrót: R. Przesuwanie: prawy przycisk myszy")
        self.retouch_row.connect("notify::active", self._on_retouch_toggled)
        self.retouch_group.add(self.retouch_row)

        brush_row = Adw.ActionRow(title="Rozmiar pędzla", subtitle="Ctrl + kółko myszy, [ ]")
        self.brush_adj = Gtk.Adjustment(value=40, lower=2, upper=600, step_increment=1)
        brush_scale = Gtk.Scale(adjustment=self.brush_adj, hexpand=True, draw_value=False,
                                valign=Gtk.Align.CENTER)
        brush_scale.set_size_request(120, -1)
        self.brush_adj.connect("value-changed", self._on_brush_changed)
        brush_row.add_suffix(brush_scale)
        self.retouch_group.add(brush_row)

        hard_row = Adw.ActionRow(title="Twardość krawędzi")
        self.hard_adj = Gtk.Adjustment(value=0.3, lower=0.0, upper=0.95, step_increment=0.05)
        hard_scale = Gtk.Scale(adjustment=self.hard_adj, hexpand=True, draw_value=False,
                               valign=Gtk.Align.CENTER)
        hard_scale.set_size_request(120, -1)
        hard_row.add_suffix(hard_scale)
        self.retouch_group.add(hard_row)

        undo_box = Gtk.Box(spacing=6, homogeneous=True, margin_top=12)
        for label, icon, action, tip in (("Cofnij", "edit-undo-symbolic", "win.undo", "Ctrl+Z"),
                                         ("Ponów", "edit-redo-symbolic", "win.redo", "Ctrl+Shift+Z")):
            undo_box.append(Gtk.Button(child=Adw.ButtonContent(icon_name=icon, label=label),
                                       action_name=action, tooltip_text=tip))
        self.retouch_group.add(undo_box)
        side.append(self.retouch_group)

        # --- obszar główny ---
        main_view = Adw.ToolbarView()
        header = Adw.HeaderBar()
        self.sidebar_btn = Gtk.ToggleButton(icon_name="sidebar-show-symbolic",
                                            tooltip_text="Pokaż/ukryj panel", active=True)
        self.sidebar_btn.bind_property("active", split, "show-sidebar",
                                       GObject.BindingFlags.BIDIRECTIONAL | GObject.BindingFlags.SYNC_CREATE)
        header.pack_start(self.sidebar_btn)

        # przełącznik Wynik / Klatka
        self.view_result_btn = Gtk.ToggleButton(label="Wynik", tooltip_text="Pokaż wynik (Tab)")
        self.view_frame_btn = Gtk.ToggleButton(label="Wybrana klatka", group=self.view_result_btn,
                                               tooltip_text="Pokaż wybraną klatkę (Tab)")
        self.view_result_btn.set_active(True)
        self.view_result_btn.connect("toggled", self._on_view_toggled)
        view_box = Gtk.Box()
        view_box.add_css_class("linked")
        view_box.append(self.view_result_btn)
        view_box.append(self.view_frame_btn)
        self.view_box = view_box
        header.set_title_widget(view_box)

        menu = Gio.Menu()
        menu.append("Skróty klawiszowe", "win.shortcuts")
        menu.append("O programie", "win.about")
        header.pack_end(Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu,
                                       tooltip_text="Menu"))
        self.save_btn = Gtk.Button(label="Zapisz", action_name="win.save",
                                   tooltip_text="Zapisz wynik (Ctrl+S)")
        header.pack_end(self.save_btn)
        self.stack_btn = Gtk.Button(label="Połącz", action_name="win.stack",
                                    tooltip_text="Dopasuj i połącz zdjęcia (Ctrl+Enter)")
        self.stack_btn.add_css_class("suggested-action")
        header.pack_end(self.stack_btn)
        main_view.add_top_bar(header)

        self.viewer = ImageViewer()
        self.viewer.on_paint = self._on_paint
        self.viewer.on_zoom_changed = self._on_zoom_changed
        self.viewer.on_brush_changed = lambda r: self.brush_adj.set_value(r)

        overlay = Gtk.Overlay(child=self.viewer)
        # zoom
        zoom_box = Gtk.Box(spacing=4, halign=Gtk.Align.END, valign=Gtk.Align.END,
                           margin_end=12, margin_bottom=12)
        zoom_box.add_css_class("toolbar")
        zoom_box.add_css_class("osd")
        zoom_box.add_css_class("zoom-box")
        self.zoom_label = Gtk.Label(label="100%", width_chars=5)
        zoom_box.append(Gtk.Button(icon_name="zoom-fit-best-symbolic", action_name="win.fit",
                                   tooltip_text="Dopasuj do okna (F)"))
        zoom_box.append(Gtk.Button(icon_name="zoom-original-symbolic", action_name="win.zoom100",
                                   tooltip_text="Rzeczywisty rozmiar 100% (1)"))
        zoom_box.append(self.zoom_label)
        overlay.add_overlay(zoom_box)
        self.zoom_box = zoom_box

        # postęp
        prog = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12, halign=Gtk.Align.CENTER,
                       valign=Gtk.Align.CENTER, visible=False)
        prog.add_css_class("card")
        prog.add_css_class("progress-card")
        self.progress_label = Gtk.Label(label="Przygotowanie…")
        self.progress_label.add_css_class("heading")
        self.progress_bar = Gtk.ProgressBar()
        self.progress_bar.set_size_request(360, -1)
        cancel = Gtk.Button(label="Anuluj", halign=Gtk.Align.CENTER)
        cancel.add_css_class("pill")
        cancel.connect("clicked", self._on_cancel)
        prog.append(self.progress_label)
        prog.append(self.progress_bar)
        prog.append(cancel)
        overlay.add_overlay(prog)
        self.progress_card = prog

        # pusty stan
        empty = Adw.StatusPage(icon_name="camera-photo-symbolic", title="Przeciągnij tutaj zdjęcia",
                               description="Seria zdjęć makro z różnymi płaszczyznami ostrości.\n"
                                           "Obsługiwane formaty: JPG, TIFF (8 i 16 bit), PNG.")
        open_btn = Gtk.Button(label="Dodaj zdjęcia", halign=Gtk.Align.CENTER,
                              action_name="win.open")
        open_btn.add_css_class("pill")
        open_btn.add_css_class("suggested-action")
        empty.set_child(open_btn)

        self.content_stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        self.content_stack.add_named(empty, "empty")
        self.content_stack.add_named(overlay, "viewer")
        main_view.set_content(self.content_stack)
        split.set_content(main_view)

        # przeciąganie plików
        drop = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop.connect("drop", self._on_drop)
        self.content_stack.add_controller(drop)

        css = Gtk.CssProvider()
        css.load_from_data(b"""
            .progress-card { padding: 24px 32px; }
            .zoom-box { padding: 4px; border-radius: 12px; }
            .thumb { border-radius: 6px; }
            row.skipped .subtitle { color: @warning_color; }
            row.skipped .thumb { opacity: 0.45; }
        """, -1)
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css,
                                                  Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

    def _setup_actions(self) -> None:
        app = self.get_application()
        actions = {
            "open": (self._on_open, ["<Ctrl>o"]),
            "clear": (self._on_clear, []),
            "stack": (self._on_stack, ["<Ctrl>Return", "<Ctrl>KP_Enter"]),
            "save": (self._on_save, ["<Ctrl>s"]),
            "undo": (self._on_undo, ["<Ctrl>z"]),
            "redo": (self._on_redo, ["<Ctrl><Shift>z", "<Ctrl>y"]),
            "fit": (lambda *_: self.viewer.fit(), ["f"]),
            "zoom100": (lambda *_: self.viewer.zoom_to(1.0 / self.get_scale_factor()), ["1"]),
            "toggle-retouch": (lambda *_: self.retouch_row.set_active(
                not self.retouch_row.get_active()), ["r"]),
            "toggle-view": (self._toggle_view, ["Tab"]),
            "brush-smaller": (lambda *_: self.brush_adj.set_value(self.brush_adj.get_value() / 1.2),
                              ["bracketleft"]),
            "brush-bigger": (lambda *_: self.brush_adj.set_value(self.brush_adj.get_value() * 1.2),
                             ["bracketright"]),
            "shortcuts": (self._on_shortcuts, ["<Ctrl>question"]),
            "about": (self._on_about, []),
        }
        self._actions = {}
        for name, (cb, accels) in actions.items():
            act = Gio.SimpleAction.new(name, None)
            act.connect("activate", cb)
            self.add_action(act)
            self._actions[name] = act
            if accels:
                app.set_accels_for_action(f"win.{name}", accels)

    # =====================================================================
    # Stan interfejsu
    # =====================================================================
    def _update_state(self) -> None:
        has_files = bool(self.paths)
        has_result = self.result is not None
        busy = self._busy
        self._actions["stack"].set_enabled(len(self.paths) >= 2 and not busy)
        self._actions["save"].set_enabled(has_result and not busy)
        self._actions["clear"].set_enabled(has_files and not busy)
        self._actions["open"].set_enabled(not busy)
        self._actions["undo"].set_enabled(bool(self.retoucher and self.retoucher.can_undo) and not busy)
        self._actions["redo"].set_enabled(bool(self.retoucher and self.retoucher.can_redo) and not busy)
        self._actions["toggle-retouch"].set_enabled(has_result and not busy)
        self.retouch_group.set_sensitive(has_result and not busy)
        self.view_box.set_visible(has_result)
        self.view_frame_btn.set_sensitive(self.selected is not None)
        self.content_stack.set_visible_child_name("viewer" if has_files else "empty")
        self.frames_group.set_sensitive(not busy)
        n = len(self.paths)
        self.frames_group.set_title(f"Klatki ({n})" if n else "Klatki")
        for i, row in enumerate(self.rows):
            row.badge.set_visible(has_result and self.retouch_row.get_active() and i == self.selected)
        title = "Focus Stack"
        if self.dirty:
            title = "• " + title
        self.set_title(title)

    def _show_error(self, error: BaseException, stage: str = "Działanie programu") -> None:
        """Okno błędu: co się stało, dlaczego, co zrobić (+ szczegóły do skopiowania)."""
        err = error if isinstance(error, StackError) else StackError.unexpected(stage, error)
        details = err.details
        if details is None and not isinstance(error, StackError):
            details = "".join(traceback.format_exception(type(error), error,
                                                         error.__traceback__))
        self._log_error(err, details)
        body = err.explanation
        if err.hints:
            body += "\n\nCo możesz zrobić:\n" + "\n".join(f"• {h}" for h in err.hints)
        d = Adw.AlertDialog(heading=err.title, body=body)
        if details:
            view = Gtk.TextView(editable=False, monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR,
                                top_margin=6, bottom_margin=6, left_margin=6, right_margin=6)
            view.get_buffer().set_text(details)
            sc = Gtk.ScrolledWindow(child=view, min_content_height=160, max_content_height=240)
            sc.add_css_class("card")
            d.set_extra_child(Gtk.Expander(label="Szczegóły techniczne", child=sc))
            d.add_response("copy", "Kopiuj szczegóły")
        d.add_response("ok", "OK")
        d.set_default_response("ok")

        def on_response(_d, r):
            if r == "copy":
                text = f"{err.title}\n\n{err.explanation}\n\n{details}"
                self.get_clipboard().set(text)
                self._toast("Skopiowano do schowka")

        d.connect("response", on_response)
        d.present(self)

    @staticmethod
    def _log_error(err: StackError, details: str | None) -> None:
        """Zapis do ~/.cache/focus-stack/bledy.log - przydatne przy zgłaszaniu problemów."""
        try:
            import datetime
            folder = os.path.join(GLib.get_user_cache_dir(), "focus-stack")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "bledy.log"), "a", encoding="utf-8") as f:
                f.write(f"\n=== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ===\n{err}\n")
                if details:
                    f.write(details + "\n")
        except OSError:
            pass

    def _toast(self, text: str, timeout: int = 3) -> None:
        self.toasts.add_toast(Adw.Toast(title=text, timeout=timeout))

    def _on_zoom_changed(self, scale: float) -> None:
        self.zoom_label.set_label(f"{round(scale * self.get_scale_factor() * 100)}%")

    # =====================================================================
    # Pliki
    # =====================================================================
    def _on_open(self, *_):
        dialog = Gtk.FileDialog(title="Wybierz zdjęcia")
        flt = Gtk.FileFilter(name="Zdjęcia (JPG, TIFF, PNG)")
        for ext in ("jpg", "jpeg", "tif", "tiff", "png"):
            flt.add_suffix(ext)
            flt.add_suffix(ext.upper())
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(flt)
        dialog.set_filters(filters)
        dialog.open_multiple(self, None, self._on_open_done)

    def _on_open_done(self, dialog, res):
        try:
            files = dialog.open_multiple_finish(res)
        except GLib.Error:
            return
        self.add_files([f.get_path() for f in files if f.get_path()])

    def _on_drop(self, target, value, x, y) -> bool:
        self.add_files([f.get_path() for f in value.get_files() if f.get_path()])
        return True

    def add_files(self, paths: list[str]) -> None:
        if self._busy:
            return
        new = []
        for p in paths:
            if os.path.isdir(p):
                new += [os.path.join(p, f) for f in sorted(os.listdir(p))
                        if imageio.is_supported(f)]
            elif imageio.is_supported(p):
                new.append(p)
        new = [p for p in new if p not in self.paths]
        if not new:
            if paths:
                self._toast("Brak nowych zdjęć w obsługiwanych formatach")
            return
        self.paths = sorted(self.paths + new, key=lambda p: os.path.basename(p).lower())
        self._rebuild_list()
        self._invalidate_result()
        if self.selected is None:
            self._select(0)
        self._update_state()
        threading.Thread(target=self._load_thumbs, args=(list(new),), daemon=True).start()

    def _rebuild_list(self) -> None:
        old = {r.path: r for r in self.rows}
        while (child := self.listbox.get_first_child()) is not None:
            self.listbox.remove(child)
        self.rows = []
        for p in self.paths:
            row = old.get(p) or FrameRow(p, self._on_remove_row)
            self.rows.append(row)
            self.listbox.append(row)

    def _load_thumbs(self, paths: list[str]) -> None:
        for p in paths:
            try:
                data = _thumbnail(p)
            except Exception:
                continue
            GLib.idle_add(self._set_thumb, p, data)

    def _set_thumb(self, path, data):
        for r in self.rows:
            if r.path == path:
                r.set_thumb(*data)
        return False

    def _on_remove_row(self, row: FrameRow) -> None:
        if self._busy:
            return
        idx = self.paths.index(row.path)
        self.paths.pop(idx)
        self._rebuild_list()
        self._invalidate_result()
        if not self.paths:
            self.selected = None
            self.viewer.set_image(None)
        else:
            self._select(min(idx, len(self.paths) - 1))
        self._update_state()

    def _on_clear(self, *_):
        self._confirm_discard(self._do_clear)

    def _do_clear(self):
        self.paths = []
        self._rebuild_list()
        self._invalidate_result()
        self.selected = None
        self.viewer.set_image(None)
        self._update_state()

    def _invalidate_result(self) -> None:
        self.result = None
        self.retoucher = None
        self.dirty = False
        self.retouch_row.set_active(False)
        self._showing = "frame"

    # =====================================================================
    # Wybór klatki i podgląd
    # =====================================================================
    def _on_row_activated(self, listbox, row) -> None:
        self._select(self.rows.index(row))
        if self.result is not None and not self.retouch_row.get_active():
            self.view_frame_btn.set_active(True)

    def _select(self, idx: int) -> None:
        self.selected = idx
        row = self.listbox.get_row_at_index(idx)
        if row is not None:
            self.listbox.select_row(row)
        self._update_state()
        if self.result is None or self._showing == "frame":
            self._show_frame()
        if self.result is not None and self.retouch_row.get_active():
            self._preload_frame(idx)

    def _show_frame(self) -> None:
        if self.selected is None:
            return
        idx = self.selected
        if self.retoucher is not None and not self._is_skipped(idx):
            if self.retoucher.has_frame(idx):
                self.viewer.set_image(self.retoucher.frame(idx))
            else:
                self._preload_frame(idx, show=True)
            return
        path = self.paths[idx]

        def work():
            try:
                img, _ = imageio.load(path)
            except Exception as e:  # noqa: BLE001
                GLib.idle_add(self._show_error, e, "Wczytywanie zdjęcia")
                return
            GLib.idle_add(self._show_loaded, idx, img)

        threading.Thread(target=work, daemon=True).start()

    def _show_loaded(self, idx, img):
        if idx == self.selected and (self.result is None or self._showing == "frame"):
            self.viewer.set_image(img)
        return False

    def _preload_frame(self, idx: int, show: bool = False) -> None:
        r = self.retoucher
        if r is None:
            return
        if self._is_skipped(idx):
            self._toast("Ta klatka została pominięta przy łączeniu (nie pasuje do reszty "
                        "serii), więc nie może być źródłem retuszu. Wybierz inną.", 6)
            return
        if r.has_frame(idx):
            if show:
                self.viewer.set_image(r.frame(idx))
            return
        self._toast(f"Wczytywanie klatki {idx + 1}…", timeout=1)

        def work():
            with self._lock:
                try:
                    r.frame(idx)
                except Exception as e:  # noqa: BLE001
                    GLib.idle_add(self._show_error, e, "Wczytywanie klatki do retuszu")
                    return
            if show:
                GLib.idle_add(self._show_after_preload, idx)

        threading.Thread(target=work, daemon=True).start()

    def _show_after_preload(self, idx):
        if self._showing == "frame" and idx == self.selected and self.retoucher:
            self.viewer.set_image(self.retoucher.frame(idx))
        return False

    def _on_view_toggled(self, btn) -> None:
        if self.result is None:
            return
        if btn.get_active():
            self._showing = "result"
            self.viewer.set_image(self.retoucher.image if self.retoucher else self.result.image)
        else:
            self._showing = "frame"
            self._show_frame()

    def _toggle_view(self, *_):
        if self.result is None:
            return
        if self.view_result_btn.get_active() and self.selected is not None:
            self.view_frame_btn.set_active(True)
        else:
            self.view_result_btn.set_active(True)

    # =====================================================================
    # Ustawienia
    # =====================================================================
    def _on_method_changed(self, row, _pspec) -> None:
        subtitles = ["Piramida - zalecana dla włosków i drobnych struktur",
                     "Mapa głębi - wierne kolory, mniej szumu; słabsza przy włoskach"]
        row.set_subtitle(subtitles[row.get_selected()])

    def _collect_settings(self) -> pipeline.Settings:
        s = pipeline.Settings()
        s.method = METHODS[self.method_row.get_selected()][0]
        s.do_align = self.align_row.get_active()
        s.local_align = s.do_align and self.local_row.get_active()
        s.crop = self.crop_row.get_active()
        return s

    # =====================================================================
    # Stackowanie
    # =====================================================================
    def _on_stack(self, *_):
        if self.dirty:
            self._confirm_discard(self._start_stack,
                                  body="Ponowne połączenie zastąpi obecny wynik razem z retuszem.")
        else:
            self._start_stack()

    def _start_stack(self):
        paths = list(self.paths)
        settings = self._collect_settings()
        self._busy = True
        self._cancel = False
        self.retouch_row.set_active(False)
        self.progress_bar.set_fraction(0)
        self.progress_label.set_label("Przygotowanie…")
        self.progress_card.set_visible(True)
        self._update_state()

        def progress(f, msg):
            GLib.idle_add(self._set_progress, f, msg)

        def work():
            try:
                res = pipeline.run(paths, settings, progress, lambda: self._cancel)
            except pipeline.Cancelled:
                GLib.idle_add(self._stack_done, None, None)
                return
            except Exception as e:  # noqa: BLE001
                GLib.idle_add(self._stack_done, None, e)
                return
            GLib.idle_add(self._stack_done, res, None)

        threading.Thread(target=work, daemon=True).start()

    def _set_progress(self, f, msg):
        self.progress_bar.set_fraction(f)
        self.progress_label.set_label(msg)
        return False

    def _on_cancel(self, *_):
        self._cancel = True
        self.progress_label.set_label("Przerywanie…")

    def _stack_done(self, res, error):
        self._busy = False
        self.progress_card.set_visible(False)
        if error:
            self._show_error(error, stage="Łączenie zdjęć")
        elif res is None:
            self._toast("Przerwano")
        else:
            self.result = res
            gui_paths = list(self.paths)

            def load_for_gui(i: int):
                # numer klatki na liście w oknie -> numer wśród faktycznie połączonych klatek
                return res.frame(res.paths.index(gui_paths[i]))

            self.retoucher = Retoucher(res.image, load_for_gui)
            self.dirty = True
            self._showing = "result"
            self.view_result_btn.set_active(True)
            self.viewer.set_image(res.image, keep_view=False)
            self._mark_skipped()
            if res.skipped:
                self._report_skipped(res)
            else:
                fallback = [w for w in res.alignment.warnings if "zapasow" in w]
                if fallback:
                    self._toast(f"Gotowe. {len(fallback)} klatek dopasowano metodą zapasową – "
                                "warto obejrzeć wynik w powiększeniu", 6)
                else:
                    self._toast("Gotowe")
        self._update_state()
        return False

    def _is_skipped(self, idx: int | None) -> bool:
        return bool(idx is not None and self.result and self.result.skipped
                    and self.paths[idx] in self.result.skipped)

    def _mark_skipped(self) -> None:
        for i, row in enumerate(self.rows):
            if self._is_skipped(i):
                row.set_subtitle("Pominięta – nie pasuje do pozostałych klatek")
                row.add_css_class("skipped")
            else:
                row.set_subtitle("")
                row.remove_css_class("skipped")

    def _report_skipped(self, res) -> None:
        reasons = {}
        for w in res.alignment.warnings:
            name, _, why = w.partition(": ")
            reasons[name] = why.replace("pominięta – ", "")
        lines = "\n".join(f"• {os.path.basename(p)} – {reasons.get(os.path.basename(p), '')}"
                          for p in res.skipped)
        total = len(res.paths) + len(res.skipped)
        d = Adw.AlertDialog(
            heading=f"Połączono {len(res.paths)} z {total} zdjęć",
            body=f"Tych zdjęć nie udało się dopasować do reszty serii, więc zostały pominięte:\n"
                 f"{lines}\n\nCo możesz zrobić:\n"
                 "• Jeśli to zdjęcia z innego ujęcia – usuń je z listy, wynik jest gotowy.\n"
                 "• Jeśli należą do serii – zwykle brakuje klatek pośrednich (zbyt duży skok "
                 "ostrości). Dodaj je i połącz ponownie.\n"
                 "• Pominięte zdjęcia są oznaczone na liście klatek.")
        d.add_response("ok", "OK")
        d.present(self)

    # =====================================================================
    # Retusz
    # =====================================================================
    def _on_retouch_toggled(self, row, _pspec) -> None:
        on = row.get_active() and self.retoucher is not None
        self.viewer.paint_mode = on
        if on:
            if self.selected is None:
                self._select(0)
            self.view_result_btn.set_active(True)
            self._preload_frame(self.selected)
            self._toast(f"Retusz z klatki {self.selected + 1}: "
                        f"{os.path.basename(self.paths[self.selected])}")
        self.viewer.queue_draw()
        self._update_state()

    def _on_brush_changed(self, adj) -> None:
        self.viewer.brush_radius = adj.get_value()
        self.viewer.queue_draw()

    def _on_paint(self, phase: str, x: float, y: float) -> None:
        r = self.retoucher
        if r is None or self.selected is None or self._showing != "result":
            return
        idx = self.selected
        if phase == "begin":
            if self._is_skipped(idx):
                self._preload_frame(idx)  # pokaże wyjaśnienie
                self._last_paint = None
                return
            if not r.has_frame(idx):
                self._toast("Chwila - klatka jeszcze się wczytuje")
                self._last_paint = None
                return
            r.begin_stroke()
            box = r.dab(idx, x, y, self.brush_adj.get_value(), self.hard_adj.get_value())
            self._last_paint = (x, y)
        elif phase == "move":
            if self._last_paint is None:
                return
            box = r.line(idx, *self._last_paint, x, y, self.brush_adj.get_value(),
                         self.hard_adj.get_value())
            self._last_paint = (x, y)
        else:
            if self._last_paint is not None:
                r.end_stroke()
                self.dirty = True
            self._last_paint = None
            self._update_state()
            return
        if box:
            self.viewer.update_region(r.image, box)

    def _on_undo(self, *_):
        if self.retoucher:
            for box in self.retoucher.undo():
                self.viewer.update_region(self.retoucher.image, box)
            self._update_state()

    def _on_redo(self, *_):
        if self.retoucher:
            for box in self.retoucher.redo():
                self.viewer.update_region(self.retoucher.image, box)
            self._update_state()

    # =====================================================================
    # Zapis
    # =====================================================================
    def _on_save(self, *_):
        if self.result is None:
            return
        first = os.path.splitext(os.path.basename(self.paths[0]))[0]
        ext = ".tif" if self.result.bits == 16 else ".jpg"
        dialog = Gtk.FileDialog(title="Zapisz wynik", initial_name=f"{first}_stack{ext}")
        folder = os.path.dirname(self.paths[0])
        if folder:
            dialog.set_initial_folder(Gio.File.new_for_path(folder))
        filters = Gio.ListStore.new(Gtk.FileFilter)
        for name, exts in (("JPG", ("jpg", "jpeg")), ("TIFF", ("tif", "tiff"))):
            f = Gtk.FileFilter(name=name)
            for e in exts:
                f.add_suffix(e)
            filters.append(f)
        dialog.set_filters(filters)
        dialog.save(self, None, self._on_save_done)

    def _on_save_done(self, dialog, res):
        try:
            f = dialog.save_finish(res)
        except GLib.Error:
            return
        path = f.get_path()
        if not os.path.splitext(path)[1].lower() in (".jpg", ".jpeg", ".tif", ".tiff"):
            path += ".jpg"
        img = (self.retoucher.image if self.retoucher else self.result.image).copy()
        bits, src = self.result.bits, self.paths[0]
        self._toast("Zapisywanie…", timeout=1)

        def work():
            try:
                imageio.save(path, img, bits, exif_source=src)
            except Exception as e:  # noqa: BLE001
                GLib.idle_add(self._show_error, e, "Zapis wyniku")
                return
            GLib.idle_add(self._saved, path)

        threading.Thread(target=work, daemon=True).start()

    def _saved(self, path):
        self.dirty = False
        self._update_state()
        self._toast(f"Zapisano: {os.path.basename(path)}")
        return False

    # =====================================================================
    # Okna dialogowe
    # =====================================================================
    def _confirm_discard(self, on_yes, body="Niezapisany wynik zostanie utracony."):
        if not self.dirty:
            on_yes()
            return
        d = Adw.AlertDialog(heading="Odrzucić wynik?", body=body)
        d.add_response("cancel", "Anuluj")
        d.add_response("discard", "Odrzuć")
        d.set_response_appearance("discard", Adw.ResponseAppearance.DESTRUCTIVE)
        d.set_default_response("cancel")
        d.connect("response", lambda _d, r: r == "discard" and on_yes())
        d.present(self)

    def _on_close_request(self, *_):
        if self.dirty:
            def close():
                self.dirty = False
                self.close()
            self._confirm_discard(close, body="Wynik nie został zapisany. Zamknąć mimo to?")
            return True
        return False

    def _on_shortcuts(self, *_):
        rows = [("Ctrl+O", "Dodaj zdjęcia"), ("Ctrl+Enter", "Połącz"), ("Ctrl+S", "Zapisz"),
                ("Tab", "Przełącz: wynik / wybrana klatka"), ("R", "Pędzel retuszu wł./wył."),
                ("[  ]", "Mniejszy / większy pędzel"), ("Ctrl+kółko", "Rozmiar pędzla"),
                ("Ctrl+Z / Ctrl+Shift+Z", "Cofnij / ponów"), ("Kółko myszy", "Powiększenie"),
                ("Prawy przycisk + przeciąganie", "Przesuwanie obrazu"),
                ("F / 1", "Dopasuj do okna / 100%")]
        grid = Gtk.Grid(column_spacing=24, row_spacing=8, margin_top=6)
        for i, (k, v) in enumerate(rows):
            kl = Gtk.Label(label=k, xalign=1)
            kl.add_css_class("monospace")
            grid.attach(kl, 0, i, 1, 1)
            grid.attach(Gtk.Label(label=v, xalign=0), 1, i, 1, 1)
        d = Adw.AlertDialog(heading="Skróty klawiszowe", extra_child=grid)
        d.add_response("ok", "OK")
        d.present(self)

    def _on_about(self, *_):
        from .. import __version__
        d = Adw.AboutDialog(application_name="Focus Stack", version=__version__,
                            comments="Dopasowanie i łączenie zdjęć makro o różnej płaszczyźnie "
                                     "ostrości.", license_type=Gtk.License.MIT_X11,
                            application_icon="camera-photo-symbolic")
        d.present(self)
