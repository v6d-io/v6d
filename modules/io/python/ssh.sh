#!/bin/sh

set -x
HOST_NAME=$1
shift
ssh ${HOST_NAME} -- "/bin/bash -c 'cat /etc/hosts || true && \
                                   source ~/.bashrc || true && \
                                   source ~/.zshrc || true && \
                                   shopt -s huponexit 2>/dev/null || true && \
                                   $*'"
