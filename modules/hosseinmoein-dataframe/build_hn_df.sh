#!/bin/bash
mkdir -p ./DataFrame/build
cd DataFrame/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ../..
