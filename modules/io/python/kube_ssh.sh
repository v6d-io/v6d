#!/bin/sh

set -x
POD_NAME=$1
shift
kubectl exec ${POD_NAME} -c engine -- "/bin/sh -c 'cat /etc/hosts > /dev/null || true && \
                                                   source ~/.bashrc || true && \
                                                   shopt -s huponexit 2>/dev/null || true && \
                                                   $*'"
