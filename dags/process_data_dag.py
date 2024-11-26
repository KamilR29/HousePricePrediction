from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from oauth2client.service_account import ServiceAccountCredentials
import gspread

# Fetch data from Google Sheets
def fetch_data_from_sheets(**kwargs):
    credentials_path = '/opt/airflow/dags/lab2-438518-d0dc8a1fcf73.json'
    scope = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(creds)

    sheet_name = 'Train Data'
    sheet = client.open(sheet_name).sheet1
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    kwargs['ti'].xcom_push(key='raw_data', value=df.to_json())

# Clean the data
def clean_data(**kwargs):
    df_json = kwargs['ti'].xcom_pull(key='raw_data')
    df = pd.read_json(df_json)

    df = df.dropna()
    df = df.drop_duplicates()

    kwargs['ti'].xcom_push(key='cleaned_data', value=df.to_json())

# Scale and save data back to Google Sheets
def scale_and_save_data(**kwargs):
    df_json = kwargs['ti'].xcom_pull(key='cleaned_data')
    df = pd.read_json(df_json)

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_columns]

    if df_numeric.empty:
        raise ValueError("No numeric columns to process.")

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=numeric_columns)

    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    df_non_numeric = df[non_numeric_columns].copy()

    # Convert datetime columns to string
    for col in df_non_numeric.columns:
        if pd.api.types.is_datetime64_any_dtype(df_non_numeric[col]):
            df_non_numeric[col] = df_non_numeric[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    df_final = pd.concat([df_scaled, df_non_numeric.reset_index(drop=True)], axis=1)

    credentials_path = '/opt/airflow/dags/lab2-438518-d0dc8a1fcf73.json'
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(creds)

    sheet_name = 'Final Processed Data'
    try:
        sheet = client.create(sheet_name)
        sheet.share('s24513@pjwstk.edu.pl', perm_type='user', role='writer')
    except:
        sheet = client.open(sheet_name).sheet1

    worksheet = sheet.get_worksheet(0)
    worksheet.clear()
    worksheet.update([df_final.columns.values.tolist()] + df_final.values.tolist())

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
}

with DAG(
    dag_id='process_data_dag',
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 11, 25),
    catchup=False,
) as dag:

    fetch_data_task = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_data_from_sheets,
        provide_context=True,
    )

    clean_data_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        provide_context=True,
    )

    scale_and_save_data_task = PythonOperator(
        task_id='scale_and_save_data',
        python_callable=scale_and_save_data,
        provide_context=True,
    )

    fetch_data_task >> clean_data_task >> scale_and_save_data_task
