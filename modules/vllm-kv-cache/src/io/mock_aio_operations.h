/** Copyright 2020-2023 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef MODULES_VLLM_KV_CACHE_SRC_IO_MOCK_AIO_OPERATIONS_H_
#define MODULES_VLLM_KV_CACHE_SRC_IO_MOCK_AIO_OPERATIONS_H_

#include <errno.h>
#include <libaio.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>
#include "vllm-kv-cache/src/io/aio_operations.h"
#include "vllm-kv-cache/src/io/error_injection.h"

// Define IO_CMD constants if not already defined
#ifndef IO_CMD_PREAD
#define IO_CMD_PREAD 0
#endif

#ifndef IO_CMD_PWRITE
#define IO_CMD_PWRITE 1
#endif

#ifndef IO_CMD_PREADV
#define IO_CMD_PREADV 7
#endif

#ifndef IO_CMD_PWRITEV
#define IO_CMD_PWRITEV 8
#endif

namespace vineyard {

namespace vllm_kv_cache {

namespace io {

// In-memory implementation of AIO operations
class MockAIOOperations : public vineyard::vllm_kv_cache::io::IAIOOperations {
 public:
  MockAIOOperations();
  ~MockAIOOperations();

  int io_setup(int maxevents, io_context_t* ctx_idp) override;
  int io_submit(io_context_t ctx_id, int64_t nr, struct iocb* ios[]) override;
  int io_getevents(io_context_t ctx_id, int64_t min_nr, int64_t nr,
                   struct io_event* events, struct timespec* timeout) override;
  int io_destroy(io_context_t ctx_id) override;

  void io_prep_pread(struct iocb* iocb, int fd, void* buf, size_t count,
                     int64_t offset);
  void io_prep_pwrite(struct iocb* iocb, int fd, void* buf, size_t count,
                      int64_t offset);

 private:
  // Internal structure to represent an AIO context
  struct AIOContext {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<struct io_event> completed_events;
    std::atomic<bool> destroyed;
    int max_events;

    explicit AIOContext(int maxevents)
        : destroyed(false), max_events(maxevents) {}
  };

  // Map of file descriptors to file data
  struct FileData {
    std::vector<char> data;
    std::mutex mutex;
  };

  std::unordered_map<int, std::shared_ptr<FileData>> files_;
  std::mutex files_mutex_;

  // Map of context IDs to context data
  std::unordered_map<io_context_t, std::shared_ptr<AIOContext>> contexts_;
  std::mutex contexts_mutex_;

  // Worker thread for processing I/O requests
  std::thread worker_thread_;
  std::atomic<bool> stop_worker_;

  // Queue for pending I/O requests
  struct PendingIO {
    io_context_t ctx_id;
    struct iocb* iocb_ptr;
  };

  std::queue<PendingIO> pending_ios_;
  std::mutex pending_ios_mutex_;
  std::condition_variable pending_ios_cv_;

  // Worker function to process I/O requests
  void ProcessIORequests();

  // Helper function to simulate I/O operation
  void SimulateIOOperation(io_context_t ctx_id, struct iocb* iocb);

  // Helper function to complete an I/O operation
  void CompleteIOOperation(io_context_t ctx_id, struct iocb* iocb,
                           ssize_t result);

  // Helper functions for file operations
  ssize_t ReadFromFile(int fd, void* buf, size_t count, off_t offset);
  ssize_t WriteToFile(int fd, const void* buf, size_t count, off_t offset);
};

}  // namespace io

}  // namespace vllm_kv_cache

}  // namespace vineyard

#endif  // MODULES_VLLM_KV_CACHE_SRC_IO_MOCK_AIO_OPERATIONS_H_
