#!/bin/bash

set -x
set -e
set -o pipefail

# python dependencies
pip3 install -U "Pygments>=2.4.1"
pip3 install -r requirements-setup.txt -r requirements.txt -r requirements-dev.txt

# linters
pip3 install black isort flake8

# build vineyard_client_python
mkdir -p build
pushd build
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DBUILD_SHARED_LIBS=ON \
         -DBUILD_VINEYARD_COVERAGE=ON \
         -DBUILD_VINEYARD_PYTHON_BINDINGS=ON \
         -DBUILD_VINEYARD_BASIC=ON \
         -DBUILD_VINEYARD_GRAPH=ON \
         -DBUILD_VINEYARD_IO=ON \
         -DBUILD_VINEYARD_IO_KAFKA=ON \
         -DBUILD_VINEYARD_HOSSEINMOEIN_DATAFRAME=ON \
         -DBUILD_VINEYARD_TESTS=ON || true
make vineyard_basic_gen -j`nproc` || true
make vineyard_client_python -j`nproc` || true
popd

# generate docs
pushd docs
make html
popd

# finish
echo "Build docs successfully! generated to docs/_build/html."
