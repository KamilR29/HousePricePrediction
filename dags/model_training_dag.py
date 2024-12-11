from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import json

# Ścieżki
processed_data_path = '/opt/airflow/processed_data/test_data.csv'
models_dir = '/opt/airflow/models'
reports_dir = '/opt/airflow/reports'
final_model_path = os.path.join(models_dir, 'final_model.pkl')
report_file_path = os.path.join(reports_dir, 'model_validation_report.json')

# Krytyczny próg dla jakości modelu
CRITICAL_THRESHOLD = 0.80

# Upewnij się, że katalogi istnieją
os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# Domyślne argumenty DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 27),
    'email': ['s24513@pjwstk.edu.pl'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}


# Funkcja do ładowania i przetwarzania danych
def load_and_preprocess_data(**kwargs):
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Plik danych testowych {processed_data_path} nie istnieje.")

    df = pd.read_csv(processed_data_path)

    # Przetwarzanie kolumn
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.drop(columns=["date", "street", "city", "statezip", "country"])

    X = df.drop(columns=["price"])
    y = df["price"]

    kwargs['ti'].xcom_push(key='X_test', value=X.to_json())
    kwargs['ti'].xcom_push(key='y_test', value=y.to_json())


# Funkcja do walidacji modelu
def validate_model(**kwargs):
    ti = kwargs['ti']
    X_test = pd.read_json(ti.xcom_pull(key='X_test'))
    y_test = pd.read_json(ti.xcom_pull(key='y_test'), typ='series')

    if not os.path.exists(final_model_path):
        raise FileNotFoundError(f"Plik modelu {final_model_path} nie istnieje.")

    # Załaduj model
    with open(final_model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Przygotowanie danych testowych
    if hasattr(model, 'steps'):
        # Obsługa pipeline'u z preprocesorem
        try:
            pipeline_features = model.steps[0][1].get_feature_names_out()
            missing_features = [feat for feat in pipeline_features if feat not in X_test.columns]
            if missing_features:
                raise ValueError(f"Brakuje kolumn w danych testowych: {missing_features}")
            X_test = X_test[pipeline_features]
        except AttributeError:
            raise ValueError("Pipeline nie zawiera metody `get_feature_names_out`.")
    else:
        # Obsługa modelu bez pipeline'u
        pass  # Używamy X_test bez zmian

    # Przewidywanie i obliczanie wskaźników jakości
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        # Generowanie raportu
        report = {
            "accuracy": accuracy,
            "mae": mae,
            "rmse": rmse,
            "critical_threshold": CRITICAL_THRESHOLD
        }
        with open(report_file_path, 'w') as report_file:
            json.dump(report, report_file, indent=4)
        print(f"Raport walidacji zapisano w: {report_file_path}")

        # Zapis wyników do XCom
        ti.xcom_push(key='model_accuracy', value=accuracy)
        ti.xcom_push(key='validation_report', value=report)

        if accuracy < CRITICAL_THRESHOLD:
            raise ValueError(f"Accuracy modelu ({accuracy:.2f}) spadło poniżej krytycznego progu {CRITICAL_THRESHOLD:.2f}.")
    except Exception as e:
        raise RuntimeError(f"Błąd podczas walidacji modelu: {str(e)}")

# Funkcja do wysyłania powiadomień
def send_notification(**kwargs):
    ti = kwargs['ti']
    report = ti.xcom_pull(key='validation_report')

    subject = "Raport Walidacji Modelu"
    body = f"""
    Walidacja modelu została zakończona.<br>
    <strong>Dokładność:</strong> {report['accuracy']}<br>
    <strong>MAE:</strong> {report['mae']}<br>
    <strong>RMSE:</strong> {report['rmse']}<br>
    <strong>Próg krytyczny:</strong> {report['critical_threshold']}<br>
    """

    return {
        'to': 's24513@pjwstk.edu.pl',
        'subject': subject,
        'html_content': body
    }


# Definicja DAG-a
with DAG(
    'model_validation_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:
    load_data_task = PythonOperator(
        task_id='load_and_preprocess_data',
        python_callable=load_and_preprocess_data,
        provide_context=True
    )

    validate_model_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
        provide_context=True
    )

    send_email_task = EmailOperator(
        task_id='send_email',
        to='s24513@pjwstk.edu.pl',
        subject='Alert: Walidacja Modelu',
        html_content='Model przeszedł walidację. Sprawdź szczegóły w raporcie.',
    )

    load_data_task >> validate_model_task >> send_email_task
