#!/bin/sh

set -x
HOST_NAME=$1
shift
ssh ${HOST_NAME} -- "/bin/bash -c 'cat /etc/hosts > /dev/null || true && \
                                   source ~/.bashrc || true && \
                                   shopt -s huponexit 2>/dev/null || true && \
                                   $*'"
