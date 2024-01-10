KV-state cache on Vineyard
=============================

Run test
--------

Build vineyard and vineyard test

.. code:: bash
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    make vineyard-test -j$(nproc)

Start vineyard server

.. code:: bash
    cd build
    ./bin/vineyardd --socket=/tmp/vineyard_test.sock  # make sure the env VINEYARD_IPC_SOCKET is set properly

Run test

.. code:: bash
    cd build
    ./bin/kv_state_cache_test
