# Focus Stack

Proste narzędzie do focus stackingu zdjęć makro na Linuksa (JPG, TIFF 8/16 bit).

Status: **krok 1 - silnik** (dopasowanie + łączenie). Interfejs graficzny w przygotowaniu.

## Jak to działa

1. **Dopasowanie (align)** - klatki są dopasowywane parami do sąsiadów (SIFT + RANSAC,
   potem dokładne ECC), z korekcją przesunięcia, obrotu i skali ("oddychanie" obiektywu)
   oraz wyrównaniem jasności. Wynik jest przycinany do obszaru wspólnego dla wszystkich klatek.
2. **Łączenie (stack)**
   - `pyramid` - odpowiednik PMax z redukcją echa (ocena ostrości w otoczeniu piksela,
     decyzje drobnych warstw wpływają na grubsze, filtr spójności),
   - `depthmap` - odpowiednik DMAP (mapa głębi z płynnymi przejściami).

## Testy

```bash
python -m tests.test_quality
```

Generuje syntetyczną serię o znanym wyniku i mierzy dokładność dopasowania oraz jakość.
