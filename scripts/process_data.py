import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autogluon.tabular import TabularPredictor
import numpy as np

# Ustawienia datasetu
dataset = 'shree1992/housedata'
file_name = 'data.csv'
data_folder = 'data'
output_folder = 'processed_data'
sample_size = 4600  # Użyj pełnego zbioru danych


# Pobieranie danych
def download_data(dataset, data_folder, file_name):
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    os.system(f'kaggle datasets download -d {dataset} -p {data_folder} --unzip')
    print(f"Pobrano plik {file_name} do folderu {data_folder}")


# Inżynieria cech
def feature_engineering(data):
    # Usunięcie niepotrzebnych kolumn
    data = data.drop(columns=['date', 'street', 'country'])

    # Tworzenie nowych cech
    data['house_age'] = 2024 - data['yr_built']
    data['years_since_renovation'] = np.where(data['yr_renovated'] == 0, 0, 2024 - data['yr_renovated'])

    # Usunięcie kolumn, które są już zastąpione
    data = data.drop(columns=['yr_built', 'yr_renovated'])

    # Transformacja logarytmiczna zmiennej docelowej
    data['price'] = np.log1p(data['price'])

    return data


# Skalowanie cech
def scale_features(train_data, val_data):
    numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()

    train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
    val_data[numeric_cols] = scaler.transform(val_data[numeric_cols])

    return train_data, val_data


# Procesowanie i podział danych
def process_data(file_path, output_folder, sample_size):
    data = pd.read_csv(file_path)
    data = feature_engineering(data)

    # Podział na zbiór treningowy i walidacyjny
    train_data, val_data = train_test_split(data, train_size=0.7, random_state=42)

    # Skalowanie cech
    train_data, val_data = scale_features(train_data, val_data)

    # Upewnienie się, że istnieje folder na wyniki
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Zapisanie danych
    train_data.to_csv(os.path.join(output_folder, 'train_data.csv'), index=False)
    val_data.to_csv(os.path.join(output_folder, 'val_data.csv'), index=False)

    print("\nPodział danych zakończony.")
    return train_data, val_data


# Trenowanie modelu
def model_selection(train_data, label_column):
    predictor = TabularPredictor(label=label_column, path="AutogluonModels", eval_metric="r2").fit(
        train_data,
        presets="best_quality",
        time_limit=3600
    )
    return predictor


# Główna funkcja
def main():
    file_path = os.path.join(data_folder, file_name)
    download_data(dataset, data_folder, file_name)

    # Procesowanie danych
    train_data, val_data = process_data(file_path, output_folder, sample_size)

    # Ustawienie kolumny docelowej
    label_column = 'price'
    predictor = model_selection(train_data, label_column)

    # Ewaluacja na zbiorze walidacyjnym
    performance = predictor.evaluate(val_data)
    print("\nWyniki modelu na zbiorze walidacyjnym:", performance)


if __name__ == "__main__":
    main()
