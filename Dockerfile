# Użyj obrazu bazowego Airflow
FROM apache/airflow:2.5.1-python3.8

# Przełącz na użytkownika root, aby zainstalować zależności systemowe i Python
# Zainstaluj systemowe pakiety jako root
USER root
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Przełącz na użytkownika airflow i zainstaluj biblioteki Python
USER airflow
RUN pip install --no-cache-dir scikit-learn pandas kaggle gspread oauth2client
