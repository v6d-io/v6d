FROM ghcr.io/v6d-io/v6d/vineyard-python-dev

# Install prerequisites
RUN apt-get update && apt-get install -y \
    curl --no-install-recommends

RUN curl -LO https://dl.k8s.io/release/v1.24.0/bin/linux/amd64/kubectl && \
    chmod +x kubectl && \
    mv kubectl /usr/bin && \
    pip3 install --no-cache-dir kubernetes torch

WORKDIR /

COPY train-data.py /

ENTRYPOINT ["python3", "train-data.py"]