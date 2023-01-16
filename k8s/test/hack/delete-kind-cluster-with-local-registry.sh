#!/bin/sh
set -o errexit

kind_name=kind

if [ ! -z "$1" ] ; then
    kind_name=$1
fi
# delete the local registry container if it's running
reg_name='kind-registry'
reg_port='5001'
if [ "$(docker inspect -f '{{.State.Running}}' "${reg_name}" 2>/dev/null || true)" == 'true' ]; then
  docker stop ${reg_name}
  docker rm ${reg_name}
fi

# delete the kind cluster
kind delete cluster --name=${kind_name} 

