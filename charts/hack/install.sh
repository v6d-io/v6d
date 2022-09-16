#!/bin/bash

set -x
set -e
set -o pipefail

ROOT=$(dirname "${BASH_SOURCE[0]}")/..

helm install vineyard-operator "$ROOT/vineyard-operator/" \
             --namespace vineyard-system \
             --create-namespace \
             
set +x
set +e
set +o pipefail
