# Blob benchmark test

In the blob benchmark test, we will focus on the performance of the basic blob
operations: 

- `PutBlob`: Put data into local vineyard. It contains two steps: first, create a blob in vineyard, then copy the data into the blob.
- `GetBlob`: Get data from local vineyard.
- `PutBlobs`: Put multiple blobs into local vineyard.
- `GetBlobs`: Get multiple blobs from local vineyard.
- `PutRemoteBlob`: Put data into remote vineyard. Unlike `PutBlob`, the data is prepared beforehand and then copied into a Vineyard blob. Therefore, this operation avoids manual memory copying.
- `GetRemoteBlob`: Get data from remote vineyard.
- `PutRemoteBlobs`: Put multiple blobs into remote vineyard.
- `GetRemoteBlobs`: Get multiple blobs from remote vineyard.

Also, the performance is measured in terms of **throughput** and **latency**.

## Build the benchmark

Configure with the following arguments when building vineyard:

```bash
cmake .. -DBUILD_VINEYARD_BENCHMARKS=ON
```

Then make the following targets:

```bash
make vineyard_benchmarks -j
```

The artifacts will be placed under the `${CMAKE_BINARY_DIR}/bin/` directory:

```bash
./bin/blob_benchmark
```

## Run the benchmark with customized parameters

**Important** Before running the benchmark, you need to start the vineyard server first. You could refer to the [launching vineyard server guide](https://v6d.io/notes/getting-started.html#launching-vineyard-server) for more information.

After that, you could get the help information by running the following command:

```bash
./bin/blob_benchmark --help
Usage: ./bin/blob_benchmark [OPTIONS]
Options:
  -h, --help                     Show this help message and exit
  -i, --ipc_socket=IPC_SOCKET    Specify the IPC socket path (required)
  -r, --rpc_endpoint=RPC_ENDPOINT  Specify the RPC endpoint (required)
  -d, --rdma_endpoint=RDMA_ENDPOINT  Specify the RDMA endpoint (required)
  -c, --clients_num=NUM          Number of clients (required)
  -s, --data_size=SIZE           Data size (e.g., 1KB, 1MB) (required)
  -n, --requests_num=NUM         Number of requests (required)
  -t, --num_threads=NUM          Number of threads (required)
  -o, --operation=TYPE           Operation type (put_blob, get_blob, put_blobs, get_blobs, put_remote_blob, get_remote_blob, put_remote_blobs, get_remote_blobs) (required)
```

For example, you could run the following command to test the performance of `PutBlob`:

```bash
./bin/blob_benchmark -i /tmp/vineyard.sock -r "127.0.0.1:9600" -d "" -c 50 -s 8MB -n 1000 -t 10 -o "put_blob"
```
