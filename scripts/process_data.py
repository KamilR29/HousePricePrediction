import os
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport

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

    return train_data, val_data


# Funkcja eksploracyjna EDA
def data_exploration(data):
    print("\nPodstawowe informacje o zbiorze danych:")
    print(data.info())

    print("\nStatystyki opisowe dla zmiennych numerycznych:")
    print(data.describe())

    print("\nAnaliza wartości brakujących:")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])

    # Histogramy dla zmiennych numerycznych
    data.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Histogramy zmiennych numerycznych")
    plt.show()

    # Wykresy pudełkowe dla zmiennych numerycznych
    for column in data.select_dtypes(include='number').columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=data[column])
        plt.title(f"Wykres pudełkowy dla {column}")
        plt.show()

    # Macierz korelacji
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Macierz korelacji")
    plt.show()


# Automatyczne generowanie raportu EDA
def generate_report(data):
    profile = ProfileReport(data, title="Raport EDA", explorative=True)
    profile.to_file("eda_report.html")
    print("Raport EDA został zapisany jako 'eda_report.html'")


# Funkcja do automatycznego doboru modeli z AutoGluon
def model_selection(train_data, label_column):
    predictor = TabularPredictor(label=label_column, path="AutogluonModels").fit(train_data)
    leaderboard = predictor.leaderboard(train_data, silent=True)
    print("\nRekomendacje modeli:")
    print(leaderboard)
    return predictor


# Główna funkcja
def main():
    file_path = os.path.join(data_folder, file_name)
    download_data(dataset, data_folder, file_name)

    # Wczytanie danych i ograniczenie do losowej próbki 5000 rekordów
    data = pd.read_csv(file_path)
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)

    # Eksploracja danych i raport EDA
    data_exploration(data)
    generate_report(data)

    # Podział na zbiór treningowy i walidacyjny
    train_data, val_data = process_data(file_path, output_folder, sample_size)

    # Automatyczny dobór modeli z AutoGluon
    label_column = 'label_column_name'  # Ustaw nazwę kolumny docelowej
    predictor = model_selection(train_data, label_column)

    # Ocena prototypowego modelu
    performance = predictor.evaluate(val_data)
    print("\nWyniki modelu na zbiorze walidacyjnym:")
    print(performance)


if __name__ == "__main__":
    main()
