# Focus Stack

Proste narzędzie do focus stackingu zdjęć makro na Linuksa (JPG, TIFF 8/16 bit).

## Instalacja (Fedora)

Jednorazowo:

```bash
git clone https://github.com/fruneman/Focus-stack.git
cd Focus-stack
./install.sh
```

Skrypt instaluje pakiety `python3-gobject gtk4 libadwaita python3-opencv python3-numpy python3-pillow`
i dodaje „Focus Stack” do menu aplikacji GNOME (oraz do „Otwórz za pomocą” dla JPG/TIFF/PNG).

## Aktualizacja

W katalogu `Focus-stack`:

```bash
git pull
./install.sh
```

Gałąź `main` zawiera zawsze sprawdzoną wersję; nowe funkcje powstają na osobnych gałęziach
i trafiają do `main` przez pull request. Numer wersji jest widoczny w menu → „O programie”.

## Obsługa

1. Przeciągnij zdjęcia do okna (lub „Dodaj zdjęcia”).
2. Kliknij **Połącz**.
3. Opcjonalnie **retusz**: wybierz na liście klatkę, w której dany fragment jest ostry i bez
   echa, włącz „Pędzel retuszu” (R) i zamaluj miejsce z echem. Tab przełącza wynik / klatkę.
   Jako **źródło pędzla** można też wybrać wynik drugiej metody (obliczany w tle z tym samym
   dopasowaniem) – np. połącz mapą głębi (czyste tło, mało szumu), a włoski i czułki domaluj
   z piramidy.
4. **Zapisz** (JPG lub TIFF; dla 16-bitowych TIFF-ów domyślnie 16-bitowy TIFF).

## Jak to działa

1. **Dopasowanie (align)** - klatki są dopasowywane parami do sąsiadów (SIFT + RANSAC,
   potem dokładne ECC), z korekcją przesunięcia, obrotu i skali ("oddychanie" obiektywu)
   oraz wyrównaniem jasności. Wynik jest przycinany do obszaru wspólnego dla wszystkich klatek.
   Gdy punkty charakterystyczne zawodzą (bardzo różna ostrość sąsiednich klatek), używana jest
   metoda zapasowa: dopasowanie rozmytych kształtów + ECC ograniczone do obszarów ze szczegółami.
   **Korekta paralaksy**: po dopasowaniu globalnym gładki przepływ optyczny lokalnie zgrywa
   obiekty leżące na różnej głębokości (główne źródło ech przy zdjęciach z ręki).
2. **Łączenie (stack)**
   - `pyramid` - odpowiednik PMax z redukcją echa (ocena ostrości w otoczeniu piksela,
     decyzje drobnych warstw wpływają na grubsze, filtr spójności),
   - `depthmap` - odpowiednik DMAP: dla każdego miejsca wybiera klatkę, w której jest ostre
     (pewność oceny = jak bardzo najostrzejsza klatka odstaje od pozostałych; niepewne miejsca
     dostają głębokość z otoczenia). Nie wzmacnia szumu ani kontrastu, wierne kolory - dobra do
     wysokiego ISO, gładkich i błyszczących powierzchni, długich serii; słabsza tam, gdzie cienkie
     struktury nakładają się na tło z innej głębokości.

## Testy

```bash
python -m tests.test_quality
python -m tests.test_retouch
python -m tests.test_align_robust
```

`test_quality` generuje syntetyczną serię o znanym wyniku i mierzy dokładność dopasowania
oraz jakość. `test_align_robust` potrzebuje prawdziwej serii testowej, która nie jest
przechowywana w `main` (oszczędność ~160 MB). Przywrócenie serii testowych z historii git:

```bash
git checkout 0b009f8 -- Stonoga-stack Pajak-stack Seria-z-bledem Seria-niedopasowana
```
