.. _gpu-memory-sharing:

Sharing GPU Memory
------------------

Vineyard supports sharing both CPU memory and GPU memory between different
processes and different compute engines. The sharing of GPU memory is archived
by using the `CUDA IPC mechanism <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaIpcMemHandle__t.html>`_
and provides a flexible unified memory interfaces.

CUDA IPC and Unified Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CUDA IPC memory handle allows GPU memory to be shared between different
processes via IPC. In vineyard, the GPU memory is allocated by the vineyardd
instance when :code:`CreateGPUBuffer()`, then an IPC handle is transferred to the
client process and the GPU memory can be accessed by the client process after
calling :code:`cudaIpcOpenMemHandle()`. For readers, the GPU memory can be accessed
like a normal CPU shared memory object with :code:`GetGPUBuffers()`.

Like `CUDA unified memory <https://developer.nvidia.com/blog/unified-memory-cuda-beginners/>`_,
vineyard's provides a unified memory interface which can be adapted to different
kinds of implementation (GPU, PPU, etc.) as the abstraction to share GPU memory
between different processes, as well as sharing memory between the host and
device.

The unified memory abstraction is able to automatically synchronize the memory
between host and devices by leverage the RAII mechanism of C++.

Example
~~~~~~~

.. note::

    The GPU shared memory is still under development and the APIs may change in
    the future.

- Creating a GPU buffer:

  .. code:: c++

      ObjectID object_id;
      Payload object;
      std::shared_ptr<MutableBuffer> buffer = nullptr;
      RETURN_ON_ERROR(client.CreateGPUBuffer(data_size(), object_id, object, buffer));

      CHECK(!buffer->is_cpu());
      CHECK(buffer->is_mutable());

  The result buffer's data :code:`buffer->mutable_data()` is a GPU memory pointer,
  which can be directly passed to GPU kernels, e.g.,

  .. code:: c++

      printKernel<<<1, 1>>>(buffer->data());

- Composing the buffer content from host code like Unified Memory:

  .. code:: c++

      {
        CUDABufferMirror mirror(*buffer, false);
        memcpy(mirror.mutable_data(), "hello world", 12);
      }

  Here the :code:`mirror`'s :code:`data()` and :code:`mutable_data()` are host memory pointers
  allocated using the :code:`cudaHostAlloc()` API. When :code:`CUDABufferMirror` destructing,
  the host memory will be copied back to the GPU memory automatically.

  The second argument of :code:`CUDABufferMirror` indicates whether the initial memory of the
  GPU buffer needs to be copied to the host memory. Defaults to :code:`false`.

- Accessing the GPU buffer from another process:

  .. code:: c++

      ObjectID object_id = ...;
      std::shared_ptr<Buffer> buffer = nullptr;
      RETURN_ON_ERROR(client.GetGPUBuffer(object_id, true, buffer));
      CHECK(!buffer->is_cpu());
      CHECK(!buffer->is_mutable());

  The result buffer's data :code:`buffer->data()` is a GPU memory pointer, which can be directly
  passed to GPU kernels, e.g.,

  .. code:: c++

      printKernel<<<1, 1>>>(buffer->data());

- Accessing the shared GPU buffer from CPU:

  .. code:: c++

      {
        CUDABufferMirror mirror(*buffer, true);
        printf("CPU data from GPU is: %s\n",
              reinterpret_cast<const char*>(mirror.data()));
      }

  Using the :code:`CUDABufferMirror` to access the GPU buffer from CPU, the mirror's :code:`data()`
  is a host memory pointer allocated using the :code:`cudaHostAlloc()` API. For immutable :code:`Buffer`,
  the second argument of :code:`CUDABufferMirror` must be :code:`true`, and the GPU memory will be
  copied to the host memory when the mirror is constructed.

- Freeing the shared GPU buffer:

  .. code:: c++

      ObjectID object_id = ...;
      RETURN_ON_ERROR(client.DelData(object_id));

For complete example about GPU memory sharing, please refer to
`gpumalloc_test.cu <https://github.com/v6d-io/v6d/blob/main/test/gpumalloc_test.cu>`_
