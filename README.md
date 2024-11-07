# Spotify Song Recommendation System

## Opis projektu
Celem projektu jest stworzenie systemu rekomendacji utworów muzycznych na podstawie ich cech, takich jak energia, taneczność, tempo i popularność. Projekt ten będzie pomocny dla platform streamingowych oraz użytkowników, którzy chcą odkrywać nową muzykę dopasowaną do ich gustu.

## Cel biznesowy/techniczny
Projekt zaspokaja potrzebę personalizacji rekomendacji w aplikacjach muzycznych, co zwiększa zaangażowanie użytkowników oraz pomaga im odkrywać nową muzykę zgodną z ich preferencjami.

## Zbiór danych
**Dataset:** "Spotify Songs Dataset" dostępny na Kaggle.
- **Źródło:** [Spotify Songs Dataset na Kaggle](https://www.kaggle.com/zaheenhamidani/ultimate-spotify-tracks-db)
- **Zawartość:** Cechy muzyczne utworów
- **Rozmiar:** Około 32 833 rekordów (zaleca się ograniczenie do mniejszych próbek)
- **Cechy numeryczne:** popularność, energia, tempo, taneczność, głośność i inne

## Zakres projektu i struktura modelu

1. **Pobieranie i przetwarzanie danych**
   - Pobranie danych z Kaggle.
   - Wstępne przetwarzanie danych: usunięcie braków, normalizacja cech.

2. **Eksploracja danych**
   - Analiza rozkładu cech i ich korelacji.
   - Wizualizacja danych i analiza wpływu cech na popularność utworów.

3. **Czyszczenie i przygotowanie danych**
   - Usunięcie brakujących danych oraz standaryzacja zmiennych numerycznych.

4. **Trenowanie modelu rekomendacji**
   - Wybór i implementacja modelu rekomendacji na podstawie cech utworów.

5. **Walidacja modelu**
   - Ocena skuteczności na zbiorze testowym przy użyciu metryk takich jak MSE lub MAE.

6. **Publikacja i wdrożenie**
   - Konteneryzacja modelu w Dockerze oraz stworzenie API do prognozowania jakości wina.

7. **Dokumentacja i prezentacja wyników**
   - Opracowanie dokumentacji oraz prezentacja wyników.

## Instrukcja uruchomienia

1. **Klonowanie repozytorium**
   ```bash
   git clone https://github.com/twojanazwa_uzytkownika/SpotifySongRecommendation.git
