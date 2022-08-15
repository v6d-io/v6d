# alloc_test

Performance testing or system benchmarks for jemalloc

Referred from <https://github.com/daanx/mimalloc-bench/blob/master/bench/alloc-test/allocator_tester.h>

## Building & run the benchmark

Configure with the following arguments when building vineyard:

```bash
cmake .. -DBUILD_VINEYARD_MALLOC=ON -DBUILD_VINEYARD_BENCHMARK=ON
```

Then make the following targets:

```bash
make vineyard_benchmarks
```

The artifacts will be placed under the `${CMAKE_BINARY_DIR}/bin/` directory:

```bash
./bin/bench_allocator_system
./bin/bench_allocator_mimalloc
./bin/bench_allocator_vineyard
```

## Run the benchmark with customized parameters

The benchmark artifacts accept an optional arguments to control the iterations count (default value is `100000000`):

```bash
./bin/bench_allocator_system 1000
./bin/bench_allocator_mimalloc 1000
./bin/bench_allocator_vineyard 1000
```

## Benchmark test

There are two kinds of allocators that we used in vineyard: jemalloc and mimalloc. You could do the following steps to reproduce the benchmark result.

### The benchmark test of mimalloc

Run the vineyard server with `8G` shared meory.

```sh
export vineyard_socket=[please create a socket file here]
./vineyardd -socket=$(vineyard_socket) -size=8G
```

Run the mimalloc benchmark test, and the argument is the iteration count.

```sh
# bench system
./bin/bench_allocator_system 1000
# bench mimalloc
./bin/bench_allocator_mimalloc 1000
# bench vineyard
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
| 10000           | 3        | nil          | 3        | 4            |
| 100000          | 3        | nil          | 3        | 4            |
| 1000000         | 9        | nil          | 8        | 10           |
