from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Define directories
raw_data_path = '/tmp/data.csv'
processed_data_dir = '/opt/airflow/processed_data'

# Ensure directories exist
os.makedirs(processed_data_dir, exist_ok=True)


def download_data():
    # Download data from Kaggle
    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    if not kaggle_user or not kaggle_key:
        raise ValueError("Kaggle API credentials are missing.")

    os.system(f"kaggle datasets download -d shree1992/housedata -p /tmp --unzip")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError("Data download failed. File not found at /tmp/data.csv")


def clean_data(**kwargs):
    # Load the raw data
    df = pd.read_csv(raw_data_path)

    # Perform data cleaning
    df = df.dropna().drop_duplicates()

    # Push cleaned data to XCom
    kwargs['ti'].xcom_push(key='cleaned_data', value=df.to_json())


def scale_split_and_save_data(**kwargs):
    # Load cleaned data from XCom
    df_json = kwargs['ti'].xcom_pull(key='cleaned_data')
    if not df_json:
        raise ValueError("No cleaned data found in XCom. Ensure the previous task executed successfully.")

    df = pd.read_json(df_json)

    # Scale numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Split data into train and test sets
    train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)

    # Ensure the `processed_data` directory exists
    os.makedirs(processed_data_dir, exist_ok=True)

    # Save train and test data to the directory
    train_data_path = os.path.join(processed_data_dir, 'train_data.csv')
    test_data_path = os.path.join(processed_data_dir, 'test_data.csv')
    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)

    print(f"Train data saved successfully to {train_data_path}")
    print(f"Test data saved successfully to {test_data_path}")


# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 27),
    'retries': 1,
}

# Define the DAG
with DAG(
        'data_processing_dag',
        default_args=default_args,
        schedule_interval=None,
        catchup=False
) as dag:
    download_data_task = PythonOperator(
        task_id='download_data',
        python_callable=download_data
    )

    clean_data_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        provide_context=True
    )

    scale_split_and_save_task = PythonOperator(
        task_id='scale_split_and_save_data',
        python_callable=scale_split_and_save_data,
        provide_context=True
    )

    download_data_task >> clean_data_task >> scale_split_and_save_task
