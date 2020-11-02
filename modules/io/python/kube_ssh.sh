#!/bin/sh

set -x
POD_NAME=$1
shift
kubectl exec ${POD_NAME} -c engine -- "/bin/sh -c 'cat /etc/hosts || true && \
                                                   source ~/.bashrc || true && \
                                                   source ~/.zshrc || true && \
                                                   $*'"
