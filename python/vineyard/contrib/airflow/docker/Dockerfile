# docker build . -f Dockerfile -t vineyardcloudnative/vineyard-airflow:2.6.3

FROM apache/airflow:2.6.3-python3.10

USER airflow

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    vineyard \
    airflow-provider-vineyard
ENV AIRFLOW__CORE__XCOM_BACKEND=vineyard.contrib.airflow.xcom.VineyardXCom
