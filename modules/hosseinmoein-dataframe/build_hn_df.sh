#!/bin/bash
mkdir -p ./thirdparty/DataFrame/build
cd thirdparty/DataFrame/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ../../..
