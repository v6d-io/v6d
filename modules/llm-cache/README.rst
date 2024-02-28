KV-state cache on Vineyard
=============================

Run test
--------

Build vineyard and vineyard test

.. code:: bash
    mkdir build
    cd build
    cmake .. -DBUILD_VINEYARD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
    make -j$(nproc)
    make vineyard_tests -j$(nproc)

Start vineyard server

.. code:: bash
    cd build
    ./bin/vineyardd --socket=/tmp/vineyard_test.sock  # make sure the env VINEYARD_IPC_SOCKET is set properly

Run test

.. code:: bash
    cd build
    export VINEYARD_IPC_SOCKET=/tmp/vineyard_test.sock
    ./bin/kv_state_cache_test
