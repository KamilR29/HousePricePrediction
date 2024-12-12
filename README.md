# House Price Prediction Project

## 1. Opis Tematu i Problemu Biznesowego

Ceny nieruchomości są kluczowym czynnikiem dla sektora nieruchomości, inwestorów, agentów oraz klientów indywidualnych. Dokładne prognozowanie cen domów pomaga nie tylko w podejmowaniu lepszych decyzji inwestycyjnych, ale także w ocenie wartości rynkowej nieruchomości. Projekt ten ma na celu stworzenie modelu predykcyjnego, który przewiduje cenę domu na podstawie różnych atrybutów, takich jak powierzchnia, liczba pokoi, lokalizacja, stan techniczny budynku oraz inne cechy fizyczne. 

Głównym celem projektu jest stworzenie prototypowego modelu predykcyjnego, który zminimalizuje błąd prognozy, pozwalając na bardziej precyzyjne szacowanie cen domów, co może wspierać decyzje zakupowe i sprzedażowe na rynku nieruchomości.

## 2. Źródło Danych i Charakterystyka

### Źródło Danych
Dane do projektu pochodzą z **[Kaggle - House Sales in King County, USA](https://www.kaggle.com/datasets/shree1992/housedata)**. Zbiór danych zawiera informacje o domach sprzedanych w regionie King County w stanie Waszyngton, USA.

### Charakterystyka Danych
Zbiór danych zawiera 4600 rekordów, z informacjami na temat sprzedanych domów, w tym następujące kolumny:

- `date`: Data sprzedaży domu.
- `price`: Cena sprzedaży domu (wartość docelowa do przewidywania).
- `bedrooms`: Liczba sypialni.
- `bathrooms`: Liczba łazienek.
- `sqft_living`: Powierzchnia mieszkalna (w stopach kwadratowych).
- `sqft_lot`: Powierzchnia działki (w stopach kwadratowych).
- `floors`: Liczba pięter.
- `waterfront`: Wskaźnik obecności widoku na wodę.
- `view`: Ocena widoku (0-4).
- `condition`: Stan techniczny budynku (1-5).
- `sqft_above`: Powierzchnia nad poziomem ziemi.
- `sqft_basement`: Powierzchnia piwnicy.
- `yr_built`: Rok budowy domu.
- `yr_renovated`: Rok ostatniego remontu.
- `street`, `city`, `statezip`, `country`: Lokalizacja nieruchomości.

### Uzasadnienie Wyboru Danych
Dane te zostały wybrane, ponieważ zawierają szczegółowe informacje o sprzedaży nieruchomości, które są niezbędne do budowy modelu predykcyjnego dla cen domów. Zawierają różnorodne cechy, które mogą mieć wpływ na cenę nieruchomości, takie jak powierzchnia, liczba pokoi, stan techniczny, rok budowy, a także lokalizacja. Dzięki tak szerokiej gamie zmiennych, model będzie mógł analizować, jakie cechy mają największy wpływ na wartość nieruchomości i dokonywać dokładnych prognoz.

## 3. Cele Projektu

Głównym celem projektu jest stworzenie modelu predykcyjnego, który przewiduje cenę domu na podstawie wybranych cech nieruchomości. Cele szczegółowe obejmują:

1. **Przygotowanie i wstępna analiza danych** – eksploracja i oczyszczenie danych w celu usunięcia wartości odstających, braków oraz sprawdzenie rozkładów cech.
2. **Budowa modelu predykcyjnego** – wykorzystanie narzędzi AutoML (takich jak TPOT) do automatycznego wyboru modeli i dostosowania hiperparametrów, aby uzyskać najlepsze możliwe wyniki.
3. **Ewaluacja i optymalizacja modelu** – porównanie wyników różnych modeli, wybranie najlepszego i jego dalsza optymalizacja.
4. **Dokumentacja i wizualizacja wyników** – przygotowanie raportu końcowego oraz wizualizacja wyników, aby zapewnić zrozumienie działania i skuteczności modelu.

new