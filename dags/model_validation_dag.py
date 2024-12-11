from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import os

# Ścieżki
test_data_path = '/opt/airflow/processed_data/test_data.csv'
model_path = '/opt/airflow/models/final_model.pkl'

# Krytyczny próg dla jakości modelu
CRITICAL_THRESHOLD = 0.80

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

# Funkcja do oceny modelu
def validate_model(**kwargs):
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Plik testowy {test_data_path} nie istnieje.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Plik modelu {model_path} nie istnieje.")

    # Załaduj dane testowe
    test_data = pd.read_csv(test_data_path)
    print(f"Kolumny danych testowych: {list(test_data.columns)}")

    # Załaduj model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Przygotowanie danych testowych
    if hasattr(model, 'steps'):
        # Obsługa pipeline'u z preprocesorem
        pipeline_features = model.steps[0][1].get_feature_names_out()
        missing_features = [feat for feat in pipeline_features if feat not in test_data.columns]
        if missing_features:
            raise ValueError(f"Brakuje kolumn w danych testowych: {missing_features}")
        X_test = test_data[pipeline_features]
    else:
        # Obsługa modelu bez pipeline'u
        X_test = test_data.drop(columns=["price"])

    y_test = test_data["price"]

    # Przewidywanie i obliczanie wskaźnika jakości
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność modelu: {accuracy:.2f}")

    # Zapis wyników do XCom
    kwargs['ti'].xcom_push(key='model_accuracy', value=accuracy)

    if accuracy < CRITICAL_THRESHOLD:
        raise ValueError(f"Accuracy modelu ({accuracy:.2f}) spadło poniżej krytycznego progu {CRITICAL_THRESHOLD:.2f}.")


# Funkcja do testów jednostkowych
def run_unit_tests(**kwargs):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        assert model is not None, "Model nie został załadowany."
    except Exception as e:
        raise AssertionError(f"Test jednostkowy nie przeszedł: {e}")

# Definicja DAG
with DAG(
        dag_id='model_validation_dag',
        default_args=default_args,
        schedule_interval=None,
        catchup=False,
) as dag:
    validate_model_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
        provide_context=True,
    )

    run_tests_task = PythonOperator(
        task_id='run_unit_tests',
        python_callable=run_unit_tests,
        provide_context=True,
    )

    send_email_task = EmailOperator(
        task_id='send_email',
        to='s24513@pjwstk.edu.pl',
        subject='Alert: Model Validation Failed',
        html_content="""
        <h3>Walidacja modelu nie powiodła się</h3>
        <p><strong>Próg krytyczny:</strong> {{ params.critical_threshold }}</p>
        <p><strong>Obecna dokładność:</strong> {{ task_instance.xcom_pull(task_ids='validate_model', key='model_accuracy') }}</p>
        """,
        params={
            'critical_threshold': CRITICAL_THRESHOLD,
        },
    )

    validate_model_task >> run_tests_task >> send_email_task
