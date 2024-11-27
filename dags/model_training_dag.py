from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tpot import TPOTRegressor
import pickle
import json

# Ścieżki katalogów
processed_data_path = '/opt/airflow/processed_data/train_data.csv'
models_dir = '/opt/airflow/models'
reports_dir = '/opt/airflow/reports'

# Upewnij się, że katalogi istnieją
os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# Domyślne argumenty DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 27),
    'retries': 1,
}


# Funkcje zadań
def load_and_preprocess_data(**kwargs):
    """Załaduj dane, przetwórz je i przygotuj do treningu."""
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Plik {processed_data_path} nie istnieje.")

    df = pd.read_csv(processed_data_path)

    # Przetwarzanie kolumn
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.drop(columns=["date", "street", "city", "statezip", "country"])

    X = df.drop(columns=["price"])  # Dopasować do rzeczywistej kolumny celu
    y = df["price"]

    # Zapis danych do XCom
    kwargs['ti'].xcom_push(key='preprocessed_X', value=X.to_json())
    kwargs['ti'].xcom_push(key='preprocessed_y', value=y.to_json())


def split_data(**kwargs):
    """Podziel dane na zbiory treningowy i testowy."""
    ti = kwargs['ti']
    X = pd.read_json(ti.xcom_pull(key='preprocessed_X'))
    y = pd.read_json(ti.xcom_pull(key='preprocessed_y'), typ='series')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ti.xcom_push(key='X_train', value=X_train.to_json())
    ti.xcom_push(key='X_test', value=X_test.to_json())
    ti.xcom_push(key='y_train', value=y_train.to_json())
    ti.xcom_push(key='y_test', value=y_test.to_json())


def train_model_with_automl(**kwargs):
    """Wytrenuj model za pomocą TPOT i zapisz go do pliku."""
    ti = kwargs['ti']
    X_train = pd.read_json(ti.xcom_pull(key='X_train'))
    y_train = pd.read_json(ti.xcom_pull(key='y_train'), typ='series')

    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42)
    tpot.fit(X_train, y_train)

    # Zapis modelu do pliku
    model_path = os.path.join(models_dir, 'trained_model.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(tpot.fitted_pipeline_, model_file)
    print(f"Model zapisano w: {model_path}")

    # Przechowaj ścieżkę modelu w XCom
    ti.xcom_push(key='trained_model_path', value=model_path)




def evaluate_model(**kwargs):
    """Ewaluacja modelu na zbiorze testowym i zapis raportu."""
    ti = kwargs['ti']
    X_test = pd.read_json(ti.xcom_pull(key='X_test'))
    y_test = pd.read_json(ti.xcom_pull(key='y_test'), typ='series')
    model_path = ti.xcom_pull(key='trained_model_path')

    # Załaduj model z pliku
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Ewaluacja modelu
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"MAE: {mae}, RMSE: {rmse}")

    # Zapis raportu
    report_path = os.path.join(reports_dir, 'evaluation_report.json')
    report_data = {
        "MAE": mae,
        "RMSE": rmse,
        "model_pipeline": str(model)
    }
    with open(report_path, "w") as report_file:
        json.dump(report_data, report_file, indent=4)
    print(f"Raport ewaluacji zapisano w: {report_path}")



def save_model(**kwargs):
    """Zapis wytrenowanego modelu do docelowego katalogu."""
    ti = kwargs['ti']
    model_path = ti.xcom_pull(key='trained_model_path')

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono pliku modelu: {model_path}")

    # Kopiowanie modelu do docelowego katalogu
    final_model_path = os.path.join(models_dir, 'final_model.pkl')
    os.rename(model_path, final_model_path)
    print(f"Model przeniesiono do: {final_model_path}")



# Definicja DAG
with DAG(
        'model_training_dag',
        default_args=default_args,
        schedule_interval=None,
        catchup=False
) as dag:
    load_and_preprocess_data_task = PythonOperator(
        task_id='load_and_preprocess_data',
        python_callable=load_and_preprocess_data,
        provide_context=True
    )

    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        provide_context=True
    )

    train_model_task = PythonOperator(
        task_id='train_model_with_automl',
        python_callable=train_model_with_automl,
        provide_context=True
    )

    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        provide_context=True
    )

    save_model_task = PythonOperator(
        task_id='save_model',
        python_callable=save_model,
        provide_context=True
    )

    # Zadania w DAG
    load_and_preprocess_data_task >> split_data_task >> train_model_task >> evaluate_model_task >> save_model_task
