from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials


# Download data from Kaggle
def download_data():
    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    if not kaggle_user or not kaggle_key:
        raise ValueError("Kaggle API credentials are missing.")
    os.system("kaggle datasets download -d shree1992/housedata -p /tmp --unzip")
    if not os.path.exists('/tmp/data.csv'):
        print("File /tmp/data.csv not found after downloading!")
    else:
        print("File /tmp/data.csv successfully downloaded!")


# Split and save data to Google Sheets
def split_and_save_data():
    # Load the downloaded data
    file_path = '/tmp/data.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError("The data file does not exist.")

    df = pd.read_csv(file_path)

    # Split data into training and testing sets
    train, test = train_test_split(df, test_size=0.3, random_state=42)

    # Save both datasets to Google Sheets
    def save_to_google_sheets(dataframe, sheet_name, credentials_path):
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
        client = gspread.authorize(creds)

        # Create or open the sheet
        try:
            sheet = client.create(sheet_name)
            sheet.share('s24513@pjwstk.edu.pl', perm_type='user', role='writer')
        except:
            sheet = client.open(sheet_name).sheet1

        worksheet = sheet.get_worksheet(0)
        worksheet.clear()
        worksheet.update([dataframe.columns.values.tolist()] + dataframe.values.tolist())

    # Define credentials and sheet names
    credentials_path = '/opt/airflow/dags/lab2-438518-d0dc8a1fcf73.json'
    save_to_google_sheets(train, 'Train Data', credentials_path)
    save_to_google_sheets(test, 'Test Data', credentials_path)


# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
}

with DAG(
        dag_id='split_data_dag',
        default_args=default_args,
        schedule_interval=None,
        start_date=datetime(2024, 11, 25),
        catchup=False,
) as dag:
    download_data_task = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
    )

    split_and_save_data_task = PythonOperator(
        task_id='split_and_save_data',
        python_callable=split_and_save_data,
    )

    download_data_task >> split_and_save_data_task
