#!/bin/bash

set -x
HOST_NAME=$1
shift

if [ "${hostname}" = "localhost" ] || [ "${hostname}" = "$(hostname)" ] || [ "${hostname}" = "$(hostname -i)" ];
then
    shopt -s huponexit 2>/dev/null || true;
    $*
else
    ssh ${HOST_NAME} -- "/bin/bash -c 'cat /etc/hosts > /dev/null || true && \
                                       source ~/.bashrc || true && \
                                       source ~/.bash_profile || true && \
                                       shopt -s huponexit 2>/dev/null || true && \
                                       $*'"
fi
