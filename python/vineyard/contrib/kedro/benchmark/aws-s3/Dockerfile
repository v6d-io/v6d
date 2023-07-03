ARG BASE_IMAGE=python:3.9-slim
FROM $BASE_IMAGE as runtime-environment

# install project requirements
COPY src/requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt
RUN pip install semver 

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

FROM runtime-environment

ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

# build the dataset
ARG DATASET_COUNT
ENV DATASET_COUNT ${DATASET_COUNT}

RUN awk 'NR == 1 {print $0; next} {for (i=1; i<=n; i++) print $0}' n=${DATASET_COUNT} data/01_raw/train.csv > temp && mv temp data/01_raw/train.csv 

CMD ["kedro", "run"]