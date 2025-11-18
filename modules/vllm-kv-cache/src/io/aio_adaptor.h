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

// Refer to: PAI-LLM/vllm/blob/develop/csrc/v6d/load_blocks_helper.hpp

#ifndef MODULES_VLLM_KV_CACHE_SRC_IO_AIO_ADAPTOR_H_
#define MODULES_VLLM_KV_CACHE_SRC_IO_AIO_ADAPTOR_H_

#include <libaio.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "common/util/callback.h"
#include "common/util/status.h"
#include "vllm-kv-cache/src/env.h"
#include "vllm-kv-cache/src/io/aio_operations.h"
#include "vllm-kv-cache/src/io/io_adaptor.h"

#include "thread-pool/thread_pool.h"

namespace vineyard {

namespace vllm_kv_cache {

namespace io {

struct AIOUserData {
  struct iocb cb;
  size_t expect_size;
  std::promise<Status> promise;
  std::atomic<bool> promise_setted{false};
  int64_t timestamp;
  std::shared_ptr<void> data_ptr;
  uint64_t enqueue_timestamp;
  uint64_t dequeue_timestamp;
  uint64_t submit_timestamp;
};

class AIOContext {
 public:
  static std::shared_ptr<AIOContext> GetSingleInstance(
      std::shared_ptr<vineyard::vllm_kv_cache::io::IAIOOperations> aio_ops) {
    static std::shared_ptr<AIOContext> instance(new AIOContext(
        std::stoi(VLLMKVCacheEnv::GetVineyardAIOSubmitConcurrency()), aio_ops));
    return instance;
  }

  std::future<Status> SubmitRead(int fd, void* data, size_t size,
                                 size_t offset);

  std::future<Status> SubmitRead(int fd, std::shared_ptr<void> data,
                                 size_t size, size_t offset);

  std::future<Status> SubmitWrite(int fd, void* data, size_t size,
                                  size_t offset);

  std::future<Status> SubmitWrite(int fd, std::shared_ptr<void> data,
                                  size_t size, size_t offset);

  Status BatchSubmitRead(int fd, std::vector<void*>& data_vec,
                         std::vector<size_t>& size_vec,
                         std::vector<size_t>& offset_vec,
                         std::vector<std::future<Status>>& results);

  Status BatchSubmitRead(int fd, std::vector<std::shared_ptr<void>>& data_vec,
                         std::vector<size_t>& size_vec,
                         std::vector<size_t>& offset_vec,
                         std::vector<std::future<Status>>& results);

  Status BatchSubmitWrite(int fd, std::vector<void*>& data_vec,
                          std::vector<size_t>& size_vec,
                          std::vector<size_t>& offset_vec,
                          std::vector<std::future<Status>>& results);

  Status BatchSubmitWrite(int fd, std::vector<std::shared_ptr<void>>& data_vec,
                          std::vector<size_t>& size_vec,
                          std::vector<size_t>& offset_vec,
                          std::vector<std::future<Status>>& results);

  size_t GetProcessingIORequest() {
    int64_t ret = submitted_requests_.load();
    return ret < 0 ? 0 : static_cast<size_t>(ret);
  }

  ~AIOContext();

 private:
  explicit AIOContext(uint64_t concurrency);

  explicit AIOContext(
      uint64_t concurrency,
      std::shared_ptr<vineyard::vllm_kv_cache::io::IAIOOperations> aio_ops);

  AIOUserData* CreateAIOUserData(size_t expect_size,
                                 std::promise<Status> promise);

  void ReleaseAIOUserData(AIOUserData* user_data);

  void PullRequest(std::vector<AIOUserData*>& user_data,
                   uint64_t queue_index = 0);

  void PushRequest(std::vector<AIOUserData*>& user_data,
                   uint64_t queue_index = 0);

  void PullIORequestThread();

  void TimeTrackerThread();

  void PushIORequestThread(uint64_t queue_index);

  uint64_t concurrency_;
  io_context_t aio_ctx_;
  std::atomic<bool> stop_{false};

  int64_t max_events_ = 32768;
  int64_t max_push_events_ = 128;
  std::atomic<uint64_t> write_counter_{0};
  std::atomic<uint64_t> read_counter_{0};
  int64_t io_timeout_milliseconds_;
  std::set<AIOUserData*> submitted_io_requests_;
  std::mutex submitted_io_requests_mutex_;

  int retry_times_ = 3;
  std::vector<std::deque<AIOUserData*>> io_queue_;
  std::vector<std::mutex> queue_mutex_;

  size_t max_pendding_requests_ = 4096;

  std::shared_ptr<ThreadPool> thread_pool_;

  std::shared_ptr<vineyard::vllm_kv_cache::io::IAIOOperations> aio_ops_;
  std::atomic<int64_t> submitted_requests_{0};
  std::vector<std::shared_ptr<std::condition_variable>> queue_cv_;
};

IOAdaptorFactory GetAIOAdaptorFactory();

class AIOAdaptor : public IIOAdaptor {
 public:
  explicit AIOAdaptor(
      const std::string& location,
      std::shared_ptr<vineyard::vllm_kv_cache::io::IAIOOperations> aio_ops =
          std::make_shared<RealAIOOperations>())
      : location_(location),
        fd_(-1),
        aio_context_(AIOContext::GetSingleInstance(aio_ops)) {}

  Status Open(std::string mode, bool direct_io = false) override;

  // not timer-safe
  Status Read(void* data, size_t size) override;

  // not timer-safe
  Status Write(void* data, size_t size) override;

  Status Close() override;

  Status GetFileSize(size_t& size) override;

  Status FileTruncate(size_t size) override;

  std::future<Status> AsyncRead(void* data, size_t size,
                                size_t offset) override;

  std::future<Status> AsyncRead(std::shared_ptr<void> data, size_t size,
                                size_t offset) override;

  std::future<Status> AsyncWrite(void* data, size_t size,
                                 size_t offset) override;

  std::future<Status> AsyncWrite(std::shared_ptr<void> data, size_t size,
                                 size_t offset) override;

  Status BatchAsyncRead(std::vector<void*>& data_vec,
                        std::vector<size_t>& size_vec,
                        std::vector<size_t>& offset_vec,
                        std::vector<std::future<Status>>& results) override;

  Status BatchAsyncWrite(std::vector<void*>& data_vec,
                         std::vector<size_t>& size_vec,
                         std::vector<size_t>& offset_vec,
                         std::vector<std::future<Status>>& results) override;

  Status BatchAsyncRead(std::vector<std::shared_ptr<void>>& data_vec,
                        std::vector<size_t>& size_vec,
                        std::vector<size_t>& offset_vec,
                        std::vector<std::future<Status>>& results) override;

  Status BatchAsyncWrite(std::vector<std::shared_ptr<void>>& data_vec,
                         std::vector<size_t>& size_vec,
                         std::vector<size_t>& offset_vec,
                         std::vector<std::future<Status>>& results) override;

  ~AIOAdaptor() {
    if (fd_ != -1) {
      Close();
    }
  }

 private:
  std::string location_;
  int fd_ = -1;
  bool direct_io_ = false;
  uint64_t read_pos = 0;
  uint64_t write_pos = 0;
  std::shared_ptr<AIOContext> aio_context_;
};

}  // namespace io

}  // namespace vllm_kv_cache

}  // namespace vineyard

#endif  // MODULES_VLLM_KV_CACHE_SRC_IO_AIO_ADAPTOR_H_
