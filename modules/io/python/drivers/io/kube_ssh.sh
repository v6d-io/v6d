#!/bin/bash

POD_NAME=$1
shift

if [[ "$POD_NAME" =~ ":" ]]; then
    KUBE_NS=${POD_NAME%%:*}
    KUBE_POD_NAME=${POD_NAME#*:}

    kubectl -n ${KUBE_NS} exec ${KUBE_POD_NAME} -c engine -- /bin/bash -c "cat /etc/hosts > /dev/null || true && \
                                                                           source ~/.bashrc || true && \
                                                                           shopt -s huponexit 2>/dev/null || true && \
                                                                           $*"
else
    kubectl exec ${POD_NAME} -c engine -- /bin/bash -c "cat /etc/hosts > /dev/null || true && \
                                                        source ~/.bashrc || true && \
                                                        shopt -s huponexit 2>/dev/null || true && \
                                                        $*"
fi
