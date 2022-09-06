# docker build . -f Dockerfile -t vineyardcloudnative/vineyard-airflow:2.3.2

FROM apache/airflow:2.3.2-python3.9

USER airflow

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    vineyard==0.6.0 \
    airflow-provider-vineyard==0.6.0
ENV AIRFLOW__CORE__XCOM_BACKEND=vineyard.contrib.airflow.xcom.VineyardXCom
