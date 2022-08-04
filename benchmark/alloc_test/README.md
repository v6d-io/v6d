# alloc_test

Performance testing or system benchmarks for jemalloc

Referred from <https://github.com/daanx/mimalloc-bench/blob/master/bench/alloc-test/allocator_tester.h>

## Building & run the benchmark

```bash
make -j$(nproc)
```

The artifacts will be placed under the `./bin/` directory:

```bash
./bin/bench_allocator_system
./bin/bench_allocator_jemalloc
./bin/bench_allocator_mimalloc
./bin/bench_allocator_vineyard
./bin/bench_allocator_vineyard_arena
```

## Build with debugging information

```bash
make -j$(nproc) DEBUG=true
```

The artifacts built with `-g -ggdb -O2` will be placed under the `./bin/` directory, with a `_dbg` suffix:

```bash
./bin/bench_allocator_system_dbg
./bin/bench_allocator_jemalloc_dbg
./bin/bench_allocator_mimalloc_dbg
./bin/bench_allocator_vineyard_dbg
./bin/bench_allocator_vineyard_arena_dbg
```

## Run the benchmark with customized parameters

The benchmark artifacts accept an optional arguments to control the iterations count (default value is `100000000`):

```bash
./bin/bench_allocator_vineyard 1000
./bin/bench_allocator_vineyard_dbg 1000
```

## Benchmark test

There are two kinds of allocators that we used in vineyard: jemalloc and mimalloc. You could do the following steps to reproduce the benchmark result.

### The benchmark test of Jemalloc

Install the jemalloc allocator in the vineyard as follows.

```sh
mkdir build
cd build
cmake .. -DBUILD_VINEYARD_MALLOC=ON -DWITH_ALLOCATOR=jemalloc
make -j8
```

Run the vineyard server with `8G` shared meory.

```sh
export vineyard_socket=[please create a socket file here]
./vineyardd -socket=$(vineyard_socket) -size=8G
```

Build the jemalloc benchmark test.

```sh
cd benchmark/alloc_test
make bench_jemalloc bench_vineyard
```

Run the jemalloc benchmark test, and the argument is the iteration count.

```sh
# the API from jemalloc
./bin/bench_allocator_jemalloc 1000
# the API from vineyard
./bin/bench_allocator_vineyard 1000
```

### The benchmark test of Mimalloc

Install the jemalloc allocator in the vineyard as follows.

```sh
mkdir build
cd build
cmake .. -DBUILD_VINEYARD_MALLOC=ON -DWITH_ALLOCATOR=mimalloc
make -j8
```

Run the vineyard server with `8G` shared meory.

```sh
export vineyard_socket=[please create a socket file here]
./vineyardd -socket=$(vineyard_socket) -size=8G
```

Build the mimalloc benchmark test.

```sh
cd benchmark/alloc_test
make bench_mimalloc bench_vineyard
```

Run the mimalloc benchmark test, and the argument is the iteration count.

```sh
# the API from mimalloc
./bin/bench_allocator_mimalloc 1000
# the API from vineyard
./bin/bench_allocator_vineyard 10000
```

### Benchmark result

The next table is benchmark test result in the machine which has 8 physical cores and 64GB of memory.

> nil means the allocator can't malloc the memory.

| iteration count | jemalloc | v6d_jemalloc | mimalloc | v6d_mimalloc |
| --------------- | -------- | ------------ | -------- | ------------ |
| 100             | 3        | 4            | 2        | 3            |
| 1000            | 3        | 9            | 3        | 4            |
| 2000            | 3        | 13           | 3        | 4            |
| 5000            | 3        | 27           | 3        | 4            |
| 10000           | 3        | nil          | 3        | 5            |
| 100000          | 3        | nil          | 3        | 12           |
| 1000000         | 9        | nil          | 8        | 92           |
