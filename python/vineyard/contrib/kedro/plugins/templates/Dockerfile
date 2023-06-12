ARG BASE_IMAGE=python:3.9-slim
FROM $BASE_IMAGE as runtime-environment

# install project requirements
COPY src/requirements.txt /tmp/requirements.txt
# delete the kedro package from requirements.txt
RUN sed -i '/kedro~=/d' /tmp/requirements.txt
RUN pip install --no-cache -r /tmp/requirements.txt && rm -f /tmp/requirements.txt
RUN pip install --no-cache-dir vineyard-kedro semver 

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
useradd -l -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

FROM runtime-environment

# copy the whole project except what is in .dockerignore
ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

RUN kedro vineyard catalog create -p __default__
