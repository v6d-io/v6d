# alloc_test

Performance testing or system benchmarks for jemalloc

Referred from https://github.com/daanx/mimalloc-bench/blob/master/bench/alloc-test/allocator_tester.h

###  Building & run the benchmark

```
make -j$(nproc)
```

The artifacts will be placed under the `./bin/` directory:

```
./bin/bench_allocator_system
./bin/bench_allocator_jemalloc
./bin/bench_allocator_vineyard
./bin/bench_allocator_vineyard_arena
```

### Build with debugging information:

```
make -j$(nproc) DEBUG=true
```

The artifacts built with `-g -ggdb -O2` will be placed under the `./bin/` directory, with a `_dbg` suffix:

```
./bin/bench_allocator_system_dbg
./bin/bench_allocator_jemalloc_dbg
./bin/bench_allocator_vineyard_dbg
./bin/bench_allocator_vineyard_arena_dbg
```

### Run the benchmark with customized parameters

The benchmark artifacts accept an optional arguments to control the iterations count (default value is `100000000`):

```
./bin/bench_allocator_vineyard 1000
./bin/bench_allocator_vineyard_dbg 1000
```
