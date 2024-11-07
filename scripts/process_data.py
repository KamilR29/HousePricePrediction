import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Ustawienie identyfikatora zestawu danych Kaggle oraz ścieżki docelowej
dataset = 'zaheenhamidani/ultimate-spotify-tracks-db'
file_name = 'SpotifyFeatures.csv'
data_folder = 'data'
output_folder = 'processed_data'
sample_size = 5000


# Funkcja do pobierania danych z Kaggle
def download_data(dataset, data_folder, file_name):
    # Upewnij się, że istnieje folder na dane
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Pobranie danych z Kaggle
    os.system(f'kaggle datasets download -d {dataset} -p {data_folder} --unzip')
    print(f"Pobrano plik {file_name} do folderu {data_folder}")


# Funkcja do przetwarzania danych: ograniczenie do 5000 rekordów i podział na zbiory
def process_data(file_path, output_folder, sample_size):
    # Wczytanie pliku CSV do DataFrame
    data = pd.read_csv(file_path)

    # Wyświetlenie podstawowych informacji o oryginalnym zbiorze danych
    print("\nPodstawowe informacje o oryginalnym zbiorze danych:")
    print(f"Liczba rekordów: {len(data)}")
    print(f"Liczba kolumn: {len(data.columns)}")
    print("\nTypy danych dla każdej kolumny:")
    print(data.dtypes)
    print("\nPodstawowe statystyki dla danych numerycznych:")
    print(data.describe())

    # Ograniczenie do 5000 losowych rekordów
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)

    # Podział na zbiór treningowy i walidacyjny (70% / 30%)
    train_data, val_data = train_test_split(data, train_size=0.7, random_state=42)

    # Upewnij się, że istnieje folder na wyniki
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Zapisanie wyników
    train_data.to_csv(os.path.join(output_folder, 'train_data.csv'), index=False)
    val_data.to_csv(os.path.join(output_folder, 'val_data.csv'), index=False)

    print("\nPodział danych zakończony.")
    print(f"Liczba rekordów w zbiorze treningowym: {len(train_data)}")
    print(f"Liczba rekordów w zbiorze walidacyjnym: {len(val_data)}")


# Główna funkcja
def main():
    file_path = os.path.join(data_folder, file_name)
    download_data(dataset, data_folder, file_name)
    process_data(file_path, output_folder, sample_size)


if __name__ == "__main__":
    main()
