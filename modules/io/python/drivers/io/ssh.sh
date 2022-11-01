#!/bin/bash

HOST_NAME=$1
shift

if [ "${HOST_NAME}" = "localhost" ] || [ "${HOST_NAME}" = "$(hostname)" ] || [ "${HOST_NAME}" = "$(hostname -i)" ];
then
    shopt -s huponexit 2>/dev/null || true;
    bash -c "$*"
else
    ssh ${HOST_NAME} -- "/bin/bash -lc '\
        cat /etc/hosts > /dev/null || true && \
        source /etc/profile || true && \
        source /etc/bash.bashrc || true && \
        source ~/.profile || true && \
        source ~/.bashrc || true && \
        source ~/.bash_profile || true && \
        shopt -s huponexit 2>/dev/null || true && \
        $*'"
fi
