import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
def load_dataset():
    os.system('kaggle datasets download -d shree1992/housedata --unzip')
    df = pd.read_csv('data.csv')
    return df

# Display basic information
def display_info(df):
    print("Number of records:", len(df))
    print("\nColumn Names and Data Types:")
    print(df.dtypes)
    print("\nMissing Values per Column:")
    print(df.isnull().sum())

# Split data into train and test sets
def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    print("\nData has been split into training and test sets.")

# Main process
if __name__ == "__main__":
    df = load_dataset()
    display_info(df)
    split_data(df)
