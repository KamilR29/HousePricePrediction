import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tpot import TPOTClassifier
import sweetviz as sv
import json

# Ustawienia do wizualizacji
sns.set(style="whitegrid")


# Download dataset from Kaggle
def download_data():
    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    if not kaggle_user or not kaggle_key:
        raise ValueError("Kaggle API credentials are missing.")
    os.system("kaggle datasets download -d shree1992/housedata -p . --unzip")


# Basic data exploration
def data_exploration(df):
    print("Basic Dataset Information:")
    print(f"Number of records: {len(df)}")
    print("\nColumn names and data types:")
    print(df.dtypes)

    missing_data = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_data[missing_data > 0])


# Analyze distributions of numerical and categorical variables
def analyze_distributions(df):
    # Histogram for numerical variables
    df.hist(bins=15, figsize=(15, 10))
    plt.savefig("histograms.png")
    plt.close()

    # Boxplots for numerical variables
    df.plot(kind="box", subplots=True, layout=(4, 4), figsize=(15, 10), sharex=False, sharey=False)
    plt.savefig("boxplots.png")
    plt.close()

    # Count plots for categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        sns.countplot(data=df, x=col)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{col}_countplot.png")
        plt.close()


# Analyze missing values
def analyze_missing_values(df):
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    print("\nMissing values per column:")
    print(missing_data)
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False)
    plt.savefig("missing_values_heatmap.png")
    plt.close()


# Correlation matrix
def correlation_matrix(df):
    # Wybierz tylko kolumny numeryczne
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.savefig("correlation_matrix.png")
    plt.close()


# Generate an automatic EDA report using Sweetviz
def generate_eda_report(df):
    report = sv.analyze(df)
    report.show_html("eda_report.html")


# Split data into training and additional training sets
def split_data(df):
    train, additional_train = train_test_split(df, test_size=0.3, random_state=42)
    print(f"Training set: {len(train)} records")
    print(f"Additional training set: {len(additional_train)} records")
    return train, additional_train


# Load and preprocess the data for model selection
def load_and_preprocess_data():
    # Przyjmujemy, że dane zostały wcześniej wczytane i przetworzone
    df = pd.read_csv("train.csv")  # Ładujemy przygotowany zbiór treningowy
    X = df.drop(columns=["price"])  # Zmienna 'price' to nasza kolumna docelowa
    y = df["price"]
    return X, y


# Perform AutoML analysis with TPOT
def perform_automl(X, y):
    # Dzielimy dane na zbiory treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicjalizacja TPOT i dopasowanie modelu
    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
    tpot.fit(X_train, y_train)

    # Wyświetlenie wyników i zapis najlepszego modelu
    print("Najlepszy model wybrany przez TPOT:")
    print(tpot.fitted_pipeline_)

    # Ocena wydajności modelu
    y_pred = tpot.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    with open("model_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Zapis najlepszej pipeline
    tpot.export("best_model_pipeline.py")


if __name__ == "__main__":
    # Step 1: Download data
    download_data()

    # Step 2: Load dataset into DataFrame
    df = pd.read_csv("data.csv")  # Adjust filename if needed

    # Step 3: Basic data exploration
    data_exploration(df)

    # Step 4: Analyze distributions
    analyze_distributions(df)

    # Step 5: Analyze missing values
    analyze_missing_values(df)

    # Step 6: Correlation matrix
    correlation_matrix(df)

    # Step 7: Generate EDA report
    generate_eda_report(df)

    # Step 8: Split data
    train, additional_train = split_data(df)

    # Optional: Save splits to CSV files if needed for further use
    train.to_csv("train.csv", index=False)
    additional_train.to_csv("additional_train.csv", index=False)

    # Step 9: Load and preprocess data for AutoML
    X, y = load_and_preprocess_data()

    # Step 10: Perform AutoML to get model recommendations
    perform_automl(X, y)
