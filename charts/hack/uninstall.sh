#!/bin/bash

ROOT=$(dirname "${BASH_SOURCE[0]}")/..

helm uninstall vineyard -n vineyard-system

kubectl -n vineyard-system delete localobjects --all 2>/dev/null || true
kubectl -n vineyard-system delete globalobjects --all 2>/dev/null || true

set +x
set +e
set +o pipefail
