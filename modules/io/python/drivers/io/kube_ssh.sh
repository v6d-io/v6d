#!/bin/bash

POD_NAME=$1
shift

KUBE_NS=""
KUBE_POD_NAME=${POD_NAME}
KUBE_CONTAINER_NAME=""
if [[ "$POD_NAME" =~ ":" ]]; then
    KUBE_NS="-n "${POD_NAME%%:*}
    KUBE_POD_CONTAINER_NAME=${POD_NAME#*:}
    if [[ "$KUBE_POD_CONTAINER_NAME" =~ ":" ]]; then
        KUBE_POD_NAME=${KUBE_POD_CONTAINER_NAME%%:*}
        KUBE_CONTAINER_NAME="-c "${KUBE_POD_CONTAINER_NAME#*:}
    else
        KUBE_POD_NAME=${KUBE_POD_CONTAINER_NAME}
    fi
fi

kubectl ${KUBE_NS} exec ${KUBE_POD_NAME} ${KUBE_CONTAINER_NAME} -- /bin/bash -lc "
        cat /etc/hosts > /dev/null || true && \
        source /etc/profile || true && \
        source /etc/bash.bashrc || true && \
        source ~/.profile || true && \
        source ~/.bashrc || true && \
        shopt -s huponexit 2>/dev/null || true && \
        $*"
