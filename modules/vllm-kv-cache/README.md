# Vineyard LLM KV Cache

## Background

Large Language Models (LLMs) are popular for their ability to generate content and solve complex tasks. However, LLM inference can be costly due to extensive GPU use and slow service engine speeds, particularly in multiple conversations. With rising demand, optimizing LLM inference throughput in multi-turn dialogues and cutting costs is crucial.

Specifically, the inference of LLM contains two phase: **Prefill** and **Decode**. The **Prefill** is to calculate the KV Cache of input tokens and the **Decode** is to generate the output tokens based on the calculated KV Cache. In multi-turn dialogues, the current input token will be superimposed with the previous output and input into the model as the new input for inference. The KV Cache of the previous input tokens can be reused in the **Prefill** phase, which can slow down the First Token Time (FTT) and improve the overall throughput.

However, the GPU memory is limited, and the KV Cache size grows linearly with the input sequence length. So we want to offload the KV Cache to the host memory or disk storage to save the GPU memory usage. We can gain benefits from the large memory capacity of host memory and disk storage to store more KV Cache data if the load cost is lower than the cost of recomputing the KV Cache.

## Design

### User defined blob with VLLMBlock object

We define a new Vineyard object named `VLLMBlock`, which represents a block of KV cache data in the vineyard blob. Each `VLLMBlock` contains multiple buffers, and each buffer corresponds to a specific layer and key/value in the LLM model.

To keep the expansibility of the kv storage, we put the memory management logic in the user side. User can use vineyard new API `GetVineyardMmapFd` to get the mmap fd of the vineyard memory, and then use the fd to mmap the memory region in the user side. Use the vllm as an example, user can implement a allocator using the same allocation strategy as vllm with the vineyard memory region, which can keep the same memory layout as vllm. Then user can implement some swap kernels to swap the GPU memory to the vineyard memory region. Then, user can use the vineyard vllm kv storage api to "register" this user allocated memory region as the vineyard blob, and create the `VLLMBlock` object to manage the offsets and sizes of each buffer in the vineyard blob. To support user defined blob, we design a new blob type named `UserBlob`, which just wraps the user allocated memory region without managing the memory allocation and deallocation. The vineyard server only manages the offsets and sizes of each buffer in the vineyard server.

Because the allocator and swap kernels are implemented in the user side, user can customize the allocation and swap strategy according to their own requirements. It also makes the vineyard vllm kv storage more flexible and adaptable to different llm engine.

We now proceed to introduce the core fields of the VLLMBlock object:

- `offsets_`: This field is a vector of offsets for each buffer in the block. Each offset indicates the distance from the mmap memory region base in the user side and the start of the buffer. The offsets are calculated based on the memory layout of the LLM model, which is typically organized by layer, key/value, and buffer. Or user can customize the memory layout according to their own requirements.

- `sizes_`: This field is a vector of sizes for each buffer in the block. Each size indicates the size of the corresponding buffer in bytes. The sizes are determined by the model configuration, such as the hidden size and data type. In most cases, the sizes of all buffers in a block are the same.

- `shape_`: This field represents the shape of the block, which is typically a 3D tensor with dimensions [layer_num, kv_num, buffer_num]. The shape indicates how many layers, keys/values, and contiguous buffers are contained in the block. It depends on the model configuration and the block size In vllm, the buffers in a block are usually discrete, so we design the shape_ to represent the logical shape of the block. Shape such as [52, 2, 1] represents a block containing 52 * 2 * 1 = 104 buffers. Each buffer is contiguous but the buffers are discrete in memory. Of course, user can implement a special swap kernel to make sure the kv cache swap to host memory in a contiguous way to make the block is a contiguous memory region, but it will increase the complexity of the swap kernel and hard to layerwise transfer because the information of layer is lost.

- `layer_index_`: This field indicates the layer index of the block in the shape_. This field is used to support flexible layout such as [kv_num, layer_num, buffer_num], which can be configured by the user according to their own requirements. Layer index is used to translate the blocks to layers for layerwise transfer( transfer between vineyardd is WIP).

As mentioned before, the vineyard vllm kv storage does not manage the memory allocation and swap logic. Instead, it relies on the user to implement these functionalities using the vineyard memory region. The vineyard vllm kv storage only provides the interface to register the user allocated memory region as the vineyard blob, and create the VLLMBlock object to manage the offsets and sizes of each buffer in the vineyard blob. This design is used to transfer between vineyardd in the future.

Because the transfer of vineyard vllm kv storage between vineyardd and prefill-decode disaggregation inference is still work in progress, now vineyard only supports the vineyard vllm kv storage as the local memory kv cache storage in a single machine.

### Disk storage as multi level cache

To further improve the capacity of the KV cache, we implement a disk storage as the multi-level cache. The vineyard llm kv cache supports all posix compliant filesystem as the backend storage, including local disk, NFS, HDFS, and other distributed filesystem.

Disk storage use multi-level directory structure to store the kv tensor files. The directory depends on the block hash value, which is calculated based on the token id. It can reduce the number of files in a single directory and improve the performance of file operations. And if the disk storage is based on distributed filesystem, it can smoothly support global kv cache in multiple machines.

Disk storage also use the AIO based asynchronous read and write to improve the performance of disk I/O. The AIO operations are implemented using the Linux native AIO library `libaio`. So this module needs to install the `libaio-dev` package before building. In informal tests, aio was able to use the whole disk bandwidth with a small amount of concurrency.

### Layerwise transfer support(WIP)

To support the transfer of kv cache between vineyardd, we design the vineyard vllm kv storage to support layerwise transfer. The layerwise transfer means that the vineyard vllm kv storage can transfer the kv cache data layer by layer. When a request arrived at the llm engine, engine processes the request layer by layer instead block by block. So the vineyard vllm kv storage can transfer the kv cache data of the required layer to the llm engine, which can overlap the transfer time and the swap time(Swap data from host memory to GPU memory).

We design a Object named VLLMLayers which represents a set of layers belongs to a set of VLLMBlocks. When engine need to fetch some remote kv cache blocks, it can request the vineyard vllm kv storage to transfer the required layers of the blocks. The vineyard vllm kv storage will find the corresponding VLLMBlocks and create a VLLMLayers object to represent the required layers. Then engine can use VLLMLayers object to check if the required layers are available in the local vineyardd or need to wait for the completion of the transfer.

The layerwise transfer implementation on the user side is completed. But the transfer between vineyardd is still work in progress. We will support the transfer of vineyard vllm kv storage between vineyardd in the future.


### Streaming support(WIP)

In the prefill-decode disaggregation inference, the kv cache is generated in the prefill phase on one machine, and then transferred to the decode phase on another machine. This scenario requires the vineyard vllm kv storage to support streaming mode, which means that the vineyard vllm kv storage can transfer the kv cache data in a streaming way. It also requires transfer layer by layer to overlap the transfer time and swap time. Stream mode is completed, but the transfer between vineyardd is still work in progress. We will support the streaming mode to transfer data between vineyardd in the future.

### IPC optimization

Because a blocks need manager a lot of small buffers. To avoid the overhead of IPC communication such as create a lot of user defined blobs, we use mmap to share the IPC msg when use the API assochiated with the vllm kv storage. The mmap fd is created when the vineyard client connect to the vineyard server. The vineyard server will reserve a large memory region for the mmap fd, and the vineyard client can use the mmap fd to mmap the memory region in the user side. Then the vineyard vllm kv storage can use the mmap memory region to store the IPC msg when use the API associated with the vllm kv storage. The implementation details are encapsulated in vllm kv storage module.
 

## Usage

We provide [C++](https://github.com/v6d-io/v6d/modules/vllm-kv-cache/src/storage/vllm_kv_storage.h) as the core API for vineyard vllm kv cache. If user implement their own block allocator and kernel swap function, they can easily use the vineyard vllm kv cache C++ API to create the vineyard blobs and VLLMBlock object to manage the kv cache data in vineyard.

We also provide a test case to show how to use the vineyard vllm kv cache C++ API. The test case is located in the [C++](https://github.com/v6d-io/v6d/modules/vllm-kv-cache/tests/vllm_storage_local_test.cc) directory. The test case shows how to create some vineyard blobs with user allocated memory region, create VLLMBlock objects to manage the offsets and sizes of each buffer in the vineyard blob, and perform basic operations such as put, get, and delete on the vineyard vllm kv cache.

### C++ API

1. First, you need to install the required dependencies.

```bash
$ cd v6d && git submodule update --init --recursive
```

2. Then, you can build the vineyard server and vineyard llm kv cache library.

```bash
$ mkdir build && cd build
$ cmake .. -DBUILD_VLLM_CACHE=ON
$ make -j
$ make vineyard_tests -j
```

After the build, you can check the `vineyardd` and `vllm_storage_local_test` in the `build/bin` directory.

```bash
$ ls build/bin
vineyardd
$ ls build/bin/vllm_storage_local_test
vllm_storage_local_test
```

3. Run the vineyard vllm kv cache test.

- Open a terminal to start the vineyard server.

```bash
$ ./build/bin/vineyardd --socket=/tmp/vineyard1.sock --meta=local --reserve_memory=true -size=4G -2M_alignment=true
```

Then open another terminal to run the vineyard llm kv cache test.

```bash
$ ./build/bin/vllm_storage_local_test
```

### Performance

Work in progress.

### Future work

- Support the transfer of vineyard vllm kv storage between vineyardd and prefill-decode disaggregation inference.
- Support the RDMA based vineyard vllm kv storage for high performance multi-node llm inference.
- Add the benchmark for vineyard vllm kv storage.