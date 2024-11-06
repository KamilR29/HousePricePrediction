# System rekomendacji książek

## Opis projektu
Celem projektu jest stworzenie systemu rekomendacji książek opartego na ocenach użytkowników i cechach książek. Model pomoże użytkownikom odkrywać nowe książki, które mogą ich zainteresować na podstawie historii ocen i preferencji.

## Cel biznesowy/techniczny
W dzisiejszych czasach personalizacja jest kluczowa dla zaangażowania użytkowników w aplikacjach do czytania i serwisach książkowych. Głównym wyzwaniem tego projektu jest zapewnienie dokładnych i spersonalizowanych rekomendacji książek, które odpowiadają różnorodnym gustom i preferencjom użytkowników. System rekomendacji pomoże spełnić tę potrzebę, zwiększając satysfakcję i zaangażowanie użytkowników.

## Zbiór danych
W projekcie wykorzystano zbiór danych [Goodbooks-10k](https://www.kaggle.com/) dostępny na Kaggle. Zbiór danych spełnia wymagania ilościowe i jakościowe projektu, zawierając ponad 10 000 książek z informacjami o ocenach użytkowników, cechach książek oraz profilach użytkowników, co umożliwia budowę modelu rekomendacji.

### Kluczowe informacje o zbiorze danych
- **Źródło:** Kaggle
- **Zawartość:** Oceny, informacje o książkach, preferencje użytkowników
- **Wielkość:** Ponad 10 000 książek, co spełnia wymaganie minimum 2000 rekordów danych tabelarycznych

## Zakres projektu i struktura modelu

1. **Pobieranie i przetwarzanie danych**
   - Import zbioru danych z Kaggle.
   - Podział danych na zbiór treningowy (70%) i walidacyjny (30%) do późniejszego doszkalania modelu.

2. **Eksploracyjna analiza danych (EDA)**
   - Przeprowadzenie analizy danych w celu zrozumienia ich struktury, zidentyfikowania braków oraz analizy podstawowych statystyk.
   - Wizualizacje i analiza wspomogą proces czyszczenia i przetwarzania danych.

3. **Trenowanie modelu rekomendacji**
   - Implementacja i trenowanie modeli rekomendacji książek.
   - Przetestowanie takich modeli, jak Collaborative Filtering oraz Content-Based Filtering, w celu znalezienia najlepszego rozwiązania.

4. **Walidacja i testowanie modelu**
   - Ocena wydajności modelu na zbiorze walidacyjnym.
   - Użycie metryk, takich jak Mean Absolute Error (MAE) oraz Mean Squared Error (MSE), do oceny jakości rekomendacji.

5. **Publikacja i wdrożenie modelu**
   - Konteneryzacja modelu za pomocą Dockera w celu ułatwienia wdrożenia.
   - Utworzenie punktu końcowego API, który umożliwi użytkownikom uzyskanie rekomendacji książek na podstawie ich danych wejściowych.

6. **Dokumentacja i prezentacja końcowa**
   - Przygotowanie pełnej dokumentacji, zawierającej szczegóły techniczne, wyniki modelu oraz kluczowe wnioski z projektu.
   - Stworzenie prezentacji końcowej, podsumowującej rezultaty i rekomendacje.

## Rozpoczęcie pracy

1. **Sklonuj to repozytorium**
   ```bash
   git clone https://github.com/twojanazwa_uzytkownika/SystemRekomendacjiKsiazek.git
