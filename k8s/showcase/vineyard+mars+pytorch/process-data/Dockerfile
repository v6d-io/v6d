FROM ghcr.io/v6d-io/v6d/vineyard-python-dev:latest

# Install prerequisites
RUN apt-get update && apt-get install -y \
    curl

RUN curl -LO https://dl.k8s.io/release/v1.24.0/bin/linux/amd64/kubectl && \
    chmod +x kubectl && \
    mv kubectl /usr/bin && \
    pip3 install --no-cache-dir \
    kubernetes pymars==0.10.0 requests urllib3

WORKDIR /

COPY process-data.py /

ENTRYPOINT ["python3", "process-data.py"]