"""Uruchamianie aplikacji okienkowej."""

from __future__ import annotations

import sys

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gio  # noqa: E402

APP_ID = "io.github.focusstack.FocusStack"


class FocusStackApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id=APP_ID,
                         flags=Gio.ApplicationFlags.HANDLES_OPEN | Gio.ApplicationFlags.NON_UNIQUE)
        self.window = None

    def _ensure_window(self):
        from .window import MainWindow
        if self.window is None:
            self.window = MainWindow(self)
        self.window.present()
        return self.window

    def do_activate(self):
        self._ensure_window()

    def do_open(self, files, n_files, hint):
        win = self._ensure_window()
        win.add_files([f.get_path() for f in files if f.get_path()])


def main() -> int:
    return FocusStackApp().run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
