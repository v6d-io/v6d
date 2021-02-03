#!/bin/bash

HOST_NAME=$1
shift

if [ "${HOST_NAME}" = "localhost" ] || [ "${HOST_NAME}" = "$(hostname)" ] || [ "${HOST_NAME}" = "$(hostname -i)" ];
then
    shopt -s huponexit 2>/dev/null || true;
    bash -c "$*"
else
    ssh ${HOST_NAME} -- "/bin/bash -c 'cat /etc/hosts > /dev/null || true && \
                                       source ~/.bashrc || true && \
                                       source ~/.bash_profile || true && \
                                       shopt -s huponexit 2>/dev/null || true && \
                                       $*'"
fi
