"""Błędy zrozumiałe dla użytkownika: co się stało, dlaczego i co można zrobić."""

from __future__ import annotations

import errno
import os
import traceback


class StackError(Exception):
    def __init__(self, title: str, explanation: str, hints: list[str] | None = None,
                 details: str | None = None):
        super().__init__(title)
        self.title = title
        self.explanation = explanation
        self.hints = hints or []
        self.details = details  # szczegóły techniczne (np. traceback) - do zgłoszenia problemu

    def __str__(self) -> str:
        parts = [self.title, self.explanation]
        if self.hints:
            parts.append("Co możesz zrobić:\n" + "\n".join(f"  • {h}" for h in self.hints))
        return "\n\n".join(parts)

    @classmethod
    def unexpected(cls, stage: str, exc: BaseException) -> "StackError":
        """Błąd, którego program nie przewidział - opisany na tyle, na ile się da."""
        if isinstance(exc, MemoryError):
            return cls(
                "Zabrakło pamięci operacyjnej",
                f"Podczas etapu „{stage}” komputerowi zabrakło pamięci RAM. Duże serie "
                "zdjęć 24 Mpx potrzebują kilku GB wolnej pamięci.",
                ["Zamknij inne programy (zwłaszcza przeglądarkę) i spróbuj ponownie.",
                 "Połącz mniejszą liczbę klatek naraz, np. usuń z listy klatki, w których "
                 "nic nie jest ostre."])
        return cls(
            f"Nieoczekiwany błąd: {stage.lower()}",
            f"Program natrafił na sytuację, której nie przewidziano, podczas etapu „{stage}”. "
            f"Komunikat techniczny: {type(exc).__name__}: {exc}",
            ["Spróbuj ponownie – jeśli błąd dotyczy dopasowania, wyłącz „Dopasuj zdjęcia” "
             "albo usuń z listy klatki, które wyraźnie odstają od reszty serii.",
             "Jeśli to nie pomaga, skopiuj szczegóły techniczne (przycisk poniżej) i przekaż je "
             "razem z informacją, na jakiej serii wystąpił błąd."],
            details="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def load_error(path: str, exc: BaseException | None = None) -> StackError:
    name = os.path.basename(path)
    if isinstance(exc, FileNotFoundError):
        return StackError(f"Nie znaleziono pliku {name}",
                          "Plik został przeniesiony, usunięty albo dysk jest odłączony.",
                          ["Usuń go z listy (ikona kosza) i dodaj zdjęcia ponownie.",
                           "Jeśli zdjęcia są na dysku zewnętrznym lub sieciowym – podłącz go."])
    if isinstance(exc, PermissionError):
        return StackError(f"Brak dostępu do pliku {name}",
                          "System nie pozwala programowi odczytać tego pliku.",
                          ["Sprawdź uprawnienia pliku (Właściwości → Uprawnienia) albo skopiuj "
                           "zdjęcia do katalogu domowego."])
    return StackError(f"Nie można wczytać pliku {name}",
                      "Plik jest uszkodzony, niekompletny (np. przerwane kopiowanie) albo ma "
                      "format, którego program nie obsługuje.",
                      ["Sprawdź, czy plik otwiera się w przeglądarce zdjęć.",
                       "Usuń go z listy (ikona kosza) i połącz pozostałe zdjęcia.",
                       "Obsługiwane formaty: JPG, TIFF (8 i 16 bit), PNG. Pliki RAW (NEF) trzeba "
                       "najpierw wywołać, np. w Darktable."],
                      details=f"{type(exc).__name__}: {exc}" if exc else None)


def save_error(path: str, exc: BaseException) -> StackError:
    name = os.path.basename(path)
    if isinstance(exc, PermissionError):
        return StackError(f"Nie można zapisać pliku {name}",
                          "Brak uprawnień do zapisu w wybranym folderze.",
                          ["Wybierz inny folder, np. Obrazy w katalogu domowym."])
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return StackError(f"Nie można zapisać pliku {name}",
                          "Na dysku zabrakło miejsca.",
                          ["Zwolnij miejsce na dysku albo zapisz w innym miejscu.",
                           "Zapis jako JPG zajmuje kilka razy mniej miejsca niż TIFF."])
    return StackError(f"Nie można zapisać pliku {name}",
                      f"Zapis się nie powiódł ({type(exc).__name__}: {exc}).",
                      ["Spróbuj zapisać w innym folderze lub w innym formacie (JPG/TIFF)."],
                      details="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
