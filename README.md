# Focus Stack

Proste narzędzie do focus stackingu zdjęć makro na Linuksa (JPG, TIFF 8/16 bit).

## Instalacja (Fedora)

Jednorazowo, w katalogu z programem:

```bash
./install.sh
```

Skrypt instaluje pakiety `python3-gobject gtk4 libadwaita python3-opencv python3-numpy python3-pillow`
i dodaje „Focus Stack” do menu aplikacji GNOME (oraz do „Otwórz za pomocą” dla JPG/TIFF/PNG).

## Obsługa

1. Przeciągnij zdjęcia do okna (lub „Dodaj zdjęcia”).
2. Kliknij **Połącz**.
3. Opcjonalnie **retusz**: wybierz na liście klatkę, w której dany fragment jest ostry i bez
   echa, włącz „Pędzel retuszu” (R) i zamaluj miejsce z echem. Tab przełącza wynik / klatkę.
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
   - `depthmap` - odpowiednik DMAP (mapa głębi z płynnymi przejściami).

## Testy

```bash
python -m tests.test_quality
python -m tests.test_retouch
python -m tests.test_align_robust
```

Generuje syntetyczną serię o znanym wyniku i mierzy dokładność dopasowania oraz jakość.
