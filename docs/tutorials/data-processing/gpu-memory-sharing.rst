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

The unified memory abstractions provides the following utilities:

- Accessing the memory from GPU or CPU using an unified abstraction: the
  :code:`GPUData()` and :code:`CPUData()`;
- Synchronizing the memory between GPU and CPU: the :code:`syncFromCPU()` and
  :code:`syncFromGPU()`.

Example
-------

.. note::

    The GPU shared memory is still under development and the APIs may change in
    the future.

- Creating a GPU buffer:

  .. code:: c++

      ObjectID object_id;
      Payload object;
      std::shared_ptr<GPUUnifiedAddress> gua = nullptr;
      RETURN_ON_ERROR(client.CreateGPUBuffer(data_size(), object_id, object, gua));

- Write data to the GPU buffer:

  .. code:: c++

      void* gpu_ptr = nullptr;
      RETURN_ON_ERROR(gua->GPUData(&gpu_ptr));
      cudaMemcpy(gpu_ptr, data, data_size(), cudaMemcpyHostToDevice);

  or, copy to CPU memory first and then synchronize to GPU memory (like CUDA unified memory):

  .. code:: c++

      void* cpu_ptr = nullptr;
      RETURN_ON_ERROR(gua->CPUData(&cpu_ptr));
      memcpy(cpu_ptr, data, data_size());
      RETURN_ON_ERROR(gua->syncFromCPU());

- Accessing the GPU buffer from another process:

  .. code:: c++

      ObjectID object_id = ...;
      std::shared_ptr<GPUUnifiedAddress> gua = nullptr;
      RETURN_ON_ERROR(client.GetGPUBuffer(object_id, true, gua));

      void* gpu_ptr = nullptr;
      RETURN_ON_ERROR(gua->GPUData(&gpu_ptr));

- Accessing the shared GPU buffer from CPU:

  .. code:: c++

      ObjectID object_id = ...;
      std::shared_ptr<GPUUnifiedAddress> gua = nullptr;
      RETURN_ON_ERROR(client.GetGPUBuffer(object_id, true, gua));

      void* cpu_ptr = nullptr;
      RETURN_ON_ERROR(gua->CPUData(&cpu_ptr));
      RETURN_ON_ERROR(gua->syncFromGPU());

- Freeing the shared GPU buffer:

  .. code:: c++

      ObjectID object_id = ...;
      RETURN_ON_ERROR(client.DelData(object_id));

:code:`UnifiedMemory` APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~

The complete :code:`UnifiedMemory` APIs are defined as:

.. code:: c++

    class GPUUnifiedAddress {
      /**
      * @brief get the cpu memry address
      *
      * @param ptr the return cpu data address
      * @return GUAError_t the error type
      */
      GUAError_t CPUData(void** ptr);

      /**
      * @brief get the gpu memory address
      *
      * @param ptr the return gpu data address
      * @return GUAError_t the error type
      */
      GUAError_t GPUData(void** ptr);

      /**
      * @brief sync data from GPU related to this gua
      *
      * @return GUAError_t the error type
      */
      GUAError_t syncFromCPU();

      /**
      * @brief sync data from CPU related to this gua
      *
      * @return GUAError_t the error type
      */
      GUAError_t syncFromGPU();

      /**
      * @brief  Malloc memory related to this gua if needed.
      *
      * @param size the memory size to be allocated
      * @param ptr the memory address on cpu or GPU
      * @param is_GPU allocate on GPU
      * @return GUAError_t the error type
      */

      GUAError_t ManagedMalloc(size_t size, void** ptr, bool is_GPU = false);
      /**
      * @brief Free the memory
      *
      */
      void ManagedFree();

      /**
      * @brief GUA to json
      *
      */
      void GUAToJSON();

      /**
      * @brief Get the Ipc Handle object
      *
      * @param handle the returned handle
      * @return GUAError_t the error type
      */

      GUAError_t getIpcHandle(cudaIpcMemHandle_t& handle);
      /**
      * @brief Set the IpcHandle of this GUA
      *
      * @param handle
      */
      void setIpcHandle(cudaIpcMemHandle_t handle);

      /**
      * @brief Get the IpcHandle of this GUA as vector
      *
      * @return std::vector<int64_t>
      */
      std::vector<int64_t> getIpcHandleVec();

      /**
      * @brief Set the IpcHandle vector of this GUA
      *
      * @param handle_vec
      */
      void setIpcHandleVec(std::vector<int64_t> handle_vec);

      /**
      * @brief Set the GPU Mem Ptr object
      *
      * @param ptr
      */
      void setGPUMemPtr(void* ptr);

      /**
      * @brief return the GPU memory pointer
      *
      * @return void* the GPU-side memory address
      */
      void* getGPUMemPtr();
      /**
      * @brief Set the Cpu Mem Ptr object
      *
      * @param ptr
      */
      void setCPUMemPtr(void* ptr);

      /**
      * @brief Get the Cpu Mem Ptr object
      *
      * @return void*
      */
      void* getCPUMemPtr();

      /**
      * @brief Get the Size object
      *
      * @return int64_t
      */
      int64_t getSize();

      /**
      * @brief Set the Size object
      *
      * @param data_size
      */
      void setSize(int64_t data_size);
    };
