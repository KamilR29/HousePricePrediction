1. Opis Projektu
Cel Projektu
Projekt Prognozowania Cen Domów ma na celu stworzenie modelu predykcyjnego, który przewiduje cenę domu na podstawie różnych atrybutów. Narzędzie to ma wspierać specjalistów ds. nieruchomości, inwestorów oraz indywidualnych nabywców w podejmowaniu świadomych decyzji poprzez dostarczanie dokładnych prognoz cen.

Dane Wykorzystane w Projekcie
Źródło: Dane pochodzą z zestawu danych Kaggle "House Sales in King County, USA."

Zawartość:

Zbór danych zawiera 4600 rekordów dotyczących sprzedaży domów w hrabstwie King, stan Waszyngton, USA.

Atrybuty obejmują:

date: Data sprzedaży

price: Cena sprzedaży (zmienna docelowa)

Cecha takie jak: bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated oraz lokalizacyjne pola.

Te atrybuty zostały wybrane w celu zapewnienia analizy różnorodnych czynników wpływających na ceny nieruchomości.

2. Opis Modelu
Wykorzystany Model
Projekt wykorzystuje pipeline uczenia maszynowego z automatycznym wyborem modelu i dostrajaniem hiperparametrów za pomocą biblioteki AutoML (np. TPOT).

Uzasadnienie Wyboru Modelu
AutoML zapewnia:

Systematyczne przeszukiwanie różnych algorytmów.

Optymalny dobór hiperparametrów.

Wybrany model minimalizuje błąd prognozy i skutecznie przetwarza dane wielowymiarowe.

Podejście to pozwala na bezproblemową adaptację do nowych danych przy minimalnej interwencji manualnej.

3. Instrukcja Użytkowania
Pobieranie Aplikacji
Sklonuj repozytorium projektu:



git clone <adres-repozytorium>
cd <folder-repozytorium>
Zainstaluj wymagane zależności:



pip install -r requirements.txt
Upewnij się, że posiadasz poświadczenia API Kaggle:

Pobierz klucz API z Kaggle (Przewodnik).

Ustaw zmienne środowiskowe dla KAGGLE_USERNAME i KAGGLE_KEY.

Uruchamianie Aplikacji
Środowisko Lokalne
Pobierz i przetwórz dane:

Przejdź do katalogu z projektem.

Uruchom DAG przetwarzania danych:



airflow dags trigger data_processing_dag
Podziel dane na zbiory treningowe i testowe:

Uruchom DAG podziału danych:



airflow dags trigger split_data_dag
Przeprowadź trenowanie i walidację modelu:

Uruchom DAG trenowania modelu:



airflow dags trigger model_training_dag
Uzyskaj wyniki:

Przetworzone zbiory danych i artefakty modelu zostaną zapisane w skonfigurowanych katalogach wyjściowych.

Korzystanie z Konfiguracji Dockerowej
Zbuduj i uruchom kontener Dockera:

Upewnij się, że pliki docker-compose.yaml i Dockerfile znajdują się w katalogu projektu.

Uruchom:



docker-compose up --build
Uzyskaj dostęp do Airflow przez interfejs webowy:

URL: http://localhost:8080

Domyślne dane logowania:

Nazwa użytkownika: admin

Hasło: admin

Uruchamiaj DAGi z poziomu interfejsu Airflow.

Wyniki i Raporty
Wyniki predykcji, raporty walidacyjne modelu oraz przetworzone dane są zapisywane w określonych katalogach wewnątrz kontenera.

Raporty mogą być również dostępne w Google Sheets, jeśli zostały skonfigurowane.

Uwagi
Upewnij się, że Twoja maszyna lub kontener Docker posiada wystarczające zasoby (RAM i CPU) do przetwarzania danych i trenowania modeli.

Zaktualizuj plik .env o niezbędne zmienne środowiskowe, jeśli korzystasz z Dockera.

4. Struktura Plików
Skrypty DAG:

data_processing_dag.py: Obsługuje pobieranie, czyszczenie i wstępne przetwarzanie danych.

split_data_dag.py: Dzieli dane na zbiory treningowe i testowe.

model_training_dag.py: Przeprowadza trenowanie i walidację modelu.

Pliki Konfiguracyjne:

docker-compose.yaml: Definiuje konteneryzowane środowisko do uruchamiania Airflow.

data_processing.yml: Definiuje pipeline CI/CD do przetwarzania i analizy danych.

Katalogi Wyjściowe:

/opt/airflow/processed_data/: Zawiera przetworzone zbiory danych.

/opt/airflow/models/: Przechowuje wytrenowane modele.

/opt/airflow/reports/: Zawiera raporty walidacyjne i wskaźniki jakości.

Dokumentacja ta zapewnia jasne zrozumienie celów projektu, danych, modelu i sposobu użycia, co ułatwia jego odtworzenie i adaptację do przyszłych zastosowań.
