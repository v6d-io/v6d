# Vineyard LLM KV Cache

## Background

Large Language Models (LLMs) are popular for their ability to generate content and solve complex tasks. However, LLM inference can be costly due to extensive GPU use and slow service engine speeds, particularly in multiple conversations. With rising demand, optimizing LLM inference throughput in multi-turn dialogues and cutting costs is crucial.

Specifically, the inference of LLM contains two phase: **Prefill** and **Decode**. The **Prefill** is to calculate the KV Cache of input tokens and the **Decode** is to generate the output tokens based on the calculated KV Cache. In multi-turn dialogues, the current input token will be superimposed with the previous output and input into the model as the new input for inference. The KV Cache of the previous input tokens can be reused in the **Prefill** phase, which can slow down the First Token Time (FTT) and improve the overall throughput.

To address the above issues, we have integrated Vineyard into LLM inference scenarios. There are currently two implementation methods: **radix tree** + **vineyard blob** and **chunk token hash** + **distributed filesystem**.

## Design

### Radix Tree + Vineyard Blob

In this method, the tokens are constructed as a radix tree and the KV tensors of these tokens are stored in Vineyard Blob (Use Memory). Also, we have some memory optimization strategies to reduce the memory usage of the radix tree such as LRU(Least Recently Used) cache and pruning.


### Token Chunk Hash + Distributed FileSystem

In this method, the tokens are chunked (e,g. 16 or 32 tokens per chunk) as a hash and the KV tensors of these tokens are stored in a distributed filesystem. Besides, we have some GC(Garbage Collection) strategies to reduce the KV tensors in the distributed filesystem.

### Comparison

In this section, we will compare the two methods in terms of latency and suitable scenarios. 

**Latency**: In a single machine, the `radix tree + vineyard blob` is faster than the `token chunk hash + distributed filesystem` method as it uses memory to store the KV tensors. When it comes to a distributed environment, the metadata synchronization from Etcd of vineyard blob will be a bottleneck.


**Suitable Scenarios**: The main factor in choosing the method is the scenario scale. If you only want to run the LLM inference in a single machine, the `radix tree + vineyard blob` method is a better choice. If you want to run the LLM inference in a distributed environment, the `token chunk hash + distributed filesystem` method is a better choice.


## Usage

We provide [C++](https://github.com/v6d-io/v6d/blob/main/modules/llm-cache/ds/kv_state_cache_manager.h) and [Python](https://github.com/v6d-io/v6d/blob/main/python/vineyard/llm/__init__.py) APIs for Vineyard LLM KV Cache. Based on the inference framework, you can use the corresponding API to integrate the Vineyard LLM KV Cache.

### C++ API

1. First, you need to install the required dependencies.

```bash
$ cd v6d && git submodule update --init --recursive
```

2. Then, you can build the vineyard server and vineyard llm kv cache library.

```bash
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DUSE_STATIC_BOOST_LIBS=OFF \
         -DBUILD_VINEYARD_SERVER=ON \
         -DBUILD_VINEYARD_CLIENT=OFF \
         -DBUILD_VINEYARD_PYTHON_BINDINGS=OFF \
         -DBUILD_VINEYARD_PYPI_PACKAGES=OFF \
         -DBUILD_VINEYARD_LLM_CACHE=ON \
         -DBUILD_VINEYARD_BASIC=OFF \
         -DBUILD_VINEYARD_GRAPH=OFF \
         -DBUILD_VINEYARD_IO=OFF \
         -DBUILD_VINEYARD_HOSSEINMOEIN_DATAFRAME=OFF \
         -DBUILD_VINEYARD_TESTS=ON \
         -DBUILD_VINEYARD_TESTS_ALL=OFF \
         -DBUILD_VINEYARD_PROFILING=OFF
$ make -j
$ make vineyard_llm_cache_tests -j
```

After the build, you can check the `vineyardd` and `libvineyard_llm_cache.so` in the `build` directory.

```bash
$ ls build/bin
vineyardd
$ ls /usr/local/lib/libvineyard_llm_cache.so
/usr/local/lib/libvineyard_llm_cache.so
```

3. Run the vineyard llm kv cache test.

- First, Build the vineyard llm kv cache test as follows.

```bash
$ cd build && make vineyard_llm_cache_tests -j
```

- Open a terminal to start the vineyard server.

```bash
$ ./build/bin/vineyardd --socket /tmp/vineyard_test.sock
```

Then open another terminal to run the vineyard llm kv cache test.

```bash
$ ./bin/kv_state_cache_test --client-num 1 --vineyard-ipc-sockets /tmp/vineyard_test.sock
```

For more information about how to use the C++ API, you can refer to the the [C++ API implementation](https://github.com/v6d-io/v6d/blob/main/modules/llm-cache/ds/kv_state_cache_manager.cc) and the [related tests](https://github.com/v6d-io/v6d/tree/main/modules/llm-cache/tests).


### Python API

1. First, same as the C++ API, you need to install the required dependencies.

```bash
$ cd v6d && git submodule update --init --recursive
```

2. Then, you can build the vineyard server and vineyard llm kv cache python
library.

```bash
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DUSE_STATIC_BOOST_LIBS=OFF \
         -DBUILD_VINEYARD_SERVER=ON \
         -DBUILD_VINEYARD_CLIENT=OFF \
         -DBUILD_VINEYARD_PYTHON_BINDINGS=ON \
         -DBUILD_VINEYARD_PYPI_PACKAGES=OFF \
         -DBUILD_VINEYARD_LLM_CACHE=ON \
         -DBUILD_VINEYARD_BASIC=OFF \
         -DBUILD_VINEYARD_GRAPH=OFF \
         -DBUILD_VINEYARD_IO=OFF \
         -DBUILD_VINEYARD_HOSSEINMOEIN_DATAFRAME=OFF \
         -DBUILD_VINEYARD_TESTS=ON \
         -DBUILD_VINEYARD_TESTS_ALL=OFF \
         -DBUILD_VINEYARD_PROFILING=OFF
$ make -j
$ make vineyard_llm_python -j
```

3. After the build, you can run the vineyard llm kv cache test as follows.

**Radix Tree + Vineyard Blob**

- Open a terminal to run the vineyard server.

```bash
$ ./build/bin/vineyardd --socket /tmp/vineyard_test.sock
```

- Open another terminal to enable the vineyard llm kv cache python module.

```bash
export PYTHONPATH=/INPUT_YOUR_PATH_HERE/v6d/python:$PYTHONPATH
```

- Then you can run the following python code to test the vineyard llm kv cache.

```python
import numpy as np
import vineyard

from vineyard.llm import KVCache
from vineyard.llm import KVTensor
from vineyard.llm.config import FileCacheConfig
from vineyard.llm.config import VineyardCacheConfig

vineyard_cache_config = VineyardCacheConfig(
    socket="/tmp/vineyard_test.sock"
    block_size=5,
    sync_interval=3,
    llm_cache_sync_lock="llmCacheSyncLock",
    llm_cache_object_name="llm_cache_object",
    llm_ref_cnt_object_name="llm_refcnt_object",
)
cache = KVCache(
    cache_config=vineyard_cache_config,
    tensor_bytes=16,  # should be the same as the nbytes of the tensor
    cache_capacity=10,
    layer=2,
)

tokens = [1, 2, 3, 4]

kv_tensors_to_update = []
kv_tensors = []
for _ in range(len(tokens)):
    k_tensor = np.random.rand(2, 2).astype(np.float32)
    v_tensor = np.random.rand(2, 2).astype(np.float32)
    kv_tensors.append([(k_tensor, v_tensor) for _ in range(cache.layer)])
    kv_tensors_to_update.append(
        [
            (
                KVTensor(k_tensor.ctypes.data, k_tensor.nbytes),
                KVTensor(v_tensor.ctypes.data, v_tensor.nbytes),
            )
            for _ in range(cache.layer)
        ]
    )

# insert the token list and the related kv cache list
updated = cache.update(None, tokens, kv_tensors_to_update)
assert updated == len(tokens)

kv_tensors_to_query = []
kv_tensors_from_cache = []
for _ in range(len(tokens)):
    kv_tensors_to_query.append(
        [
            (
                KVTensor(0, 0),
                KVTensor(0, 0),
            )
            for _ in range(cache.layer)
        ]
    )

matched = cache.query(tokens, kv_tensors_to_query)
kv_tensors_from_cache = kv_tensors_to_query[:matched]
assert matched == len(tokens)

assert len(kv_tensors) == len(kv_tensors_from_cache)
for kv, kv_from_cache in zip(kv_tensors, kv_tensors_from_cache):
    assert len(kv) == len(kv_from_cache)
    for (k_tensor, v_tensor), (queried_k_tensor, queried_v_tensor) in zip(
        kv, kv_from_cache
    ):
        queried_k_tensor = np.frombuffer(
            queried_k_tensor,
            dtype=k_tensor.dtype,
        ).reshape(k_tensor.shape)
        queried_v_tensor = np.frombuffer(
            queried_v_tensor,
            dtype=v_tensor.dtype,
        ).reshape(v_tensor.shape)
        assert np.array_equal(k_tensor, queried_k_tensor)
        assert np.array_equal(v_tensor, queried_v_tensor)
```

**Token Chunk Hash + Distributed FileSystem**

Same as previous step, you need to enable the vineyard llm kv cache python module.

```bash
$ export PYTHONPATH=/INPUT_YOUR_PATH_HERE/v6d/python:$PYTHONPATH
```

- Then you can the following python code to run the vineyard llm kv cache test.

```python
import numpy as np
import vineyard

from vineyard.llm import KVCache
from vineyard.llm import KVTensor
from vineyard.llm.config import FileCacheConfig
from vineyard.llm.config import VineyardCacheConfig

file_cache_config = FileCacheConfig(
    chunk_size=2,
    split_number=2,
    root="/tmp/vineyard/llm_cache",
)
cache = KVCache(
    cache_config=file_cache_config,
    tensor_bytes=16,  # should be the same as the nbytes of the tensor
    cache_capacity=10,
    layer=2,
)

tokens = [1, 2, 3, 4]
original_kv_tensors = []
for i in range(0, len(tokens), file_cache_config.chunk_size):
    kv_tensors_to_update = []
    k_tensor = np.random.rand(2, 2).astype(np.float32)
    v_tensor = np.random.rand(2, 2).astype(np.float32)
    for _ in range(file_cache_config.chunk_size):
        original_kv_tensors.append(
            [(k_tensor, v_tensor) for _ in range(cache.layer)]
        )
        kv_tensors_to_update.append(
            [
                (
                    KVTensor(k_tensor.ctypes.data, k_tensor.nbytes),
                    KVTensor(v_tensor.ctypes.data, v_tensor.nbytes),
                )
                for _ in range(cache.layer)
            ]
        )
    updated = cache.update(
        tokens[:i],
        tokens[i : i + file_cache_config.chunk_size],
        kv_tensors_to_update,
    )
    assert updated == file_cache_config.chunk_size

kv_tensors_from_cache = []
kv_tensors = []
for _ in range(len(tokens)):
    k_tensor = np.empty((2, 2), dtype=np.float32)
    v_tensor = np.empty((2, 2), dtype=np.float32)
    kv_tensors_from_cache.append([(k_tensor, v_tensor) for _ in range(cache.layer)])
    kv_tensors.append(
        [
            (
                KVTensor(k_tensor.ctypes.data, k_tensor.nbytes),
                KVTensor(v_tensor.ctypes.data, v_tensor.nbytes),
            )
            for _ in range(cache.layer)
        ]
    )
matched = cache.query(tokens, kv_tensors)
assert matched == len(tokens)

assert len(kv_tensors) == len(kv_tensors_from_cache)
for kv, kv_from_cache in zip(original_kv_tensors, kv_tensors_from_cache):
    assert len(kv) == len(kv_from_cache)
    for (k_tensor, v_tensor), (queried_k_tensor, queried_v_tensor) in zip(
        kv, kv_from_cache
    ):
        np.array_equal(k_tensor, queried_k_tensor)
        np.array_equal(v_tensor, queried_v_tensor)
```

After running the above code, you can check the KV Tensor file under the directory `/tmp/vineyard/llm_cache` as follows.

```bash
$ ls /tmp/vineyard/llm_cache
44  c3  __temp
```

### Performance

We have conducted some performance tests on the `Token Chunk Hash + Distributed FileSystem`.
The test environment includes the local SSD and distributed FS.

**Based on SSD**

The max read throughput of SSD is around 3GiB/s, the max write throughput of SSD is around 1.5GiB/s. Based on the machine, we can get the performance of vineyard llm kv cache as follows.

| query (token/s) | update (token/s) |
|-----------------|------------------|
|      605        |        324       |

The kv tensor size of a token is around 5MB, and the throughput is as follows.

|  query (MiB/s)  |  update (MiB/s)  |
|-----------------|------------------|
|  605 * 5 = 3025 | 324 * 5 = 1620   |


**Based on DFS**

We use the [Aliyun CPFS](https://www.aliyun.com/product/nas_cpfs) as the dfs in the benchmark test. The max write throughput of CPFS is around 20GB/s, and the max read throughput is 40GB/s. Based on the CPFS, we test the throughput of fio with multiple
worker, which can be regarded as a CPFS client.

| worker | write (MiB/s) | read (MiB/s) | CPFS aggregate bandwidth (write/read) |
|--------|---------------|--------------|---------------------------------------|
|    1   |     1315      |     2016     |              1315 / 2016              |
|    2   |     1175      |     1960     |              2360 / 3920              |
|    4   |      928      |     1780     |              3712 / 7120              |
|    8   |      895      |     1819     |              7160 / 14552             |
|   16   |      638      |     1609     |             10208 / 25744             |
|   32   |      586      |     1308     |             18752 / 41856             |

We test the vineyard llm kv cache with 32 workers, and the throughput of a single worker
is as follows.

| query (token/s) | update (token/s) |
|-----------------|------------------|
|      375        |        252       |

Same as the SSD, the kv tensor size of a token is around 5MB, and the throughput is as follows.

|  query (MiB/s)  |  update (MiB/s)  |
|-----------------|------------------|
|  375 * 5 = 1875 | 252 * 5 = 1260   |

### Conclusion

`Radix Tree + Vineyard Blob` is highly affected by the synchronization of the metadata from Etcd, which is a bottleneck in the distributed environment. In the future, we can leverage the RDMA to support fast remote read/write and reduce the synchronization cost of the metadata with new architecture such as Master-Slave.

`Token Chunk Hash + Distributed Filesystem` can make full use of the bandwidth of SSD and DFS, which can ensure that the overall inference throughput is improved at a lower SLO.

### Future work

- Support the RDMA.
- Create multiple replicas of an object in different instances, which can serve read request concurrently.
- Implement a load balancer to balance the burden of different vineyardd instances and the requests from the clients.