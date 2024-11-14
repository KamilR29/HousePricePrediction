import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tpot import TPOTRegressor
import sweetviz as sv
import json
from operator import itemgetter

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
    df.hist(bins=15, figsize=(15, 10))
    plt.savefig("histograms.png")
    plt.close()

    df.plot(kind="box", subplots=True, layout=(4, 4), figsize=(15, 10), sharex=False, sharey=False)
    plt.savefig("boxplots.png")
    plt.close()

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
    df = pd.read_csv("train.csv")

    # Przekształcenie kolumny 'date' na rok, miesiąc, dzień
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.drop(columns=["date", "street", "city", "statezip", "country"])  # Usuwamy kolumny tekstowe

    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y


# Perform AutoML analysis with TPOT and display top 3 models
def perform_automl(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicjalizacja TPOT i dopasowanie modelu
    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42)
    tpot.fit(X_train, y_train)

    # Zbieranie wyników dla wszystkich testowanych modeli
    model_scores = []
    for pipeline_string, pipeline in tpot.evaluated_individuals_.items():
        score = pipeline.get("internal_cv_score", float('-inf'))  # Pobierz wynik CV dla modelu
        model_scores.append((pipeline_string, score))

    # Sortowanie modeli według wyników walidacji krzyżowej (od najwyższego do najniższego)
    top_models = sorted(model_scores, key=itemgetter(1), reverse=True)[:3]

    # Wyświetlanie trzech najlepszych modeli
    print("Trzy najlepsze modele wybrane przez TPOT:")
    for i, (model, score) in enumerate(top_models, 1):
        print(f"{i}. Model: {model} | Wynik CV: {score}")

    # Wybór najlepszego model
