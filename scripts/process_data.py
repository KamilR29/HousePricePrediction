import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Download dataset from Kaggle
def download_data():
    # Ensure Kaggle API credentials are set up
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


# Split data into training and additional training sets
def split_data(df):
    train, additional_train = train_test_split(df, test_size=0.3, random_state=42)
    print(f"Training set: {len(train)} records")
    print(f"Additional training set: {len(additional_train)} records")
    return train, additional_train


if __name__ == "__main__":
    # Step 1: Download data
    download_data()

    # Step 2: Load dataset into DataFrame
    df = pd.read_csv("data.csv")  # Adjust filename if needed

    # Step 3: Explore data
    data_exploration(df)

    # Step 4: Split data
    train, additional_train = split_data(df)

    # Optional: Save splits to CSV files if needed for further use
    train.to_csv("train.csv", index=False)
    additional_train.to_csv("additional_train.csv", index=False)
