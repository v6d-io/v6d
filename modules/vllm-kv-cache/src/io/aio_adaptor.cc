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

// Refer to: PAI-LLM/vllm/blob/develop/csrc/v6d/load_blocks_helper.cpp

#include <fcntl.h>
#include <libaio.h>
#include <unistd.h>

#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/util/logging.h"
#include "vllm-kv-cache/src/env.h"
#include "vllm-kv-cache/src/io/aio_adaptor.h"
#include "vllm-kv-cache/src/vllm_kv_cache_util.h"

#include "thread-pool/thread_pool.h"

namespace vineyard {

namespace vllm_kv_cache {

namespace io {

AIOContext::AIOContext(uint64_t concurrency)
    : AIOContext(concurrency, std::make_shared<RealAIOOperations>()) {}

AIOContext::AIOContext(uint64_t concurrency,
                       std::shared_ptr<IAIOOperations> aio_ops)
    : concurrency_(concurrency),
      io_queue_(concurrency),
      queue_mutex_(concurrency),
      aio_ops_(std::move(aio_ops)) {
  // init context
  LOG(INFO) << "Initializing AIO context with concurrency: " << concurrency_;
  thread_pool_ = std::make_shared<ThreadPool>(concurrency_ + 2);
  for (size_t i = 0; i < concurrency_; ++i) {
    queue_cv_.push_back(std::make_shared<std::condition_variable>());
  }
  aio_ctx_ = nullptr;
  int ret = aio_ops_->io_setup(max_events_, &aio_ctx_);
  if (ret < 0) {
    throw std::runtime_error("Failed to initialize io_setup: error code:" +
                             std::to_string(ret));
  }
  io_timeout_milliseconds_ =
      std::stoll(VLLMKVCacheEnv::GetVineyardVLLMKVCacheIOTimeoutMilliseconds());
  LOG(INFO) << "AIOContext initialized with max_events: " << max_events_
            << ", io_timeout_milliseconds_: " << io_timeout_milliseconds_;
  LOG(INFO) << "Running polling thread...";

  // run polling thread
  thread_pool_->enqueue_noreturn([this]() { this->PullIORequestThread(); });

  LOG(INFO) << "Running timeout tracker thread...";
  thread_pool_->enqueue_noreturn([this]() { this->TimeTrackerThread(); });

  LOG(INFO) << "Running " << concurrency_ << " submit threads...";
  for (size_t i = 0; i < concurrency_; ++i) {
    thread_pool_->enqueue_noreturn(
        [this, i]() { this->PushIORequestThread(i); });
  }
  LOG(INFO) << "Init done";
}

void AIOContext::PullIORequestThread() {
  struct io_event* events = new io_event[max_events_];
  struct timespec timeout = {0, 1000000};  // 1ms timeout
  while (!stop_.load()) {
    usleep(stoi(VLLMKVCacheEnv::GetAIOPullResultInterval()));
    // pulling events
    int num_events =
        aio_ops_->io_getevents(aio_ctx_, 0, max_events_, events, &timeout);
    if (num_events < 0) {
      // handle error (e.g., EINTR interrupt)
      LOG(INFO) << "io_getevents error: " << strerror(-num_events);
      continue;
    }

    if (num_events == 0) {
      continue;  // no events, continue polling
    }

    submitted_requests_.fetch_sub(num_events);

    // process each completed event
    for (int i = 0; i < num_events; i++) {
      auto* user_data = static_cast<AIOUserData*>(events[i].data);

      std::lock_guard<std::mutex> lock(submitted_io_requests_mutex_);
      try {
        bool promise_set = user_data->promise_setted.exchange(true);
        if (!promise_set) {
          if (events[i].res >= 0 && events[i].res2 == 0) {  // success
            size_t bytes_transferred = events[i].res;
            if (bytes_transferred < user_data->expect_size) {
              user_data->promise.set_value(Status::EndOfFile());
            } else if (bytes_transferred > user_data->expect_size) {
              user_data->promise.set_value(
                  Status::Invalid("return size larger than expect size"));
            } else {
              user_data->promise.set_value(Status::OK());
            }
          } else {
            LOG(WARNING) << "Promise already set for AIO operation, skipping.";
          }
        }
      } catch (const std::future_error& e) {
        LOG(WARNING) << "Failed to set promise value: " << e.what()
                     << " maybe set by timeout tracker.";
      }
      submitted_io_requests_.erase(user_data);
      ReleaseAIOUserData(user_data);
    }
  }
  delete events;
  LOG(INFO) << "Polling thread stopped.";
}

void AIOContext::TimeTrackerThread() {
  // timeout checker.
  while (!stop_.load()) {
    usleep(std::stoi(
        VLLMKVCacheEnv::AIOGCWaitTimeMicroseconds()));  // sleep for 10ms
    std::lock_guard<std::mutex> lock(submitted_io_requests_mutex_);
    for (auto it = submitted_io_requests_.begin();
         it != submitted_io_requests_.end();) {
      auto user_data = *it;
      if (user_data->timestamp + io_timeout_milliseconds_ <
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count()) {
        // timeout, set promise to error
        try {
          bool promise_set = user_data->promise_setted.exchange(true);
          if (!promise_set) {
            user_data->promise.set_value(
                Status::IOError("AIO operation timed out after " +
                                std::to_string(io_timeout_milliseconds_) +
                                " ms."
                                "IO context enqueue time: " +
                                KVCacheHelper::MicrosecondToTimestamp(
                                    user_data->enqueue_timestamp) +
                                ", dequeue time: " +
                                KVCacheHelper::MicrosecondToTimestamp(
                                    user_data->dequeue_timestamp) +
                                ", submit time: " +
                                KVCacheHelper::MicrosecondToTimestamp(
                                    user_data->submit_timestamp)));
          } else {
            LOG(WARNING) << "Promise already set for AIO operation, skipping.";
          }
        } catch (const std::future_error& e) {
          LOG(WARNING) << "Failed to set timeout promise value: " << e.what();
        }
        submitted_io_requests_.erase(it++);
      } else {
        ++it;
      }
    }
  }
  LOG(INFO) << "Timeout tracker thread stopped.";
}

void AIOContext::PushIORequestThread(uint64_t i) {
  while (!stop_.load()) {
    std::vector<AIOUserData*> user_data_vec;
    PullRequest(user_data_vec, i);
    if (user_data_vec.empty()) {
      continue;
    }

    int64_t submit_requests_num = user_data_vec.size();
    std::shared_ptr<struct iocb*[]> iocbs(
        new struct iocb*[submit_requests_num]);
    for (int64_t i = 0; i < submit_requests_num; ++i) {
      iocbs[i] = &user_data_vec[i]->cb;
      user_data_vec[i]->timestamp =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
    }

    VLOG(2) << "Submitting " << submit_requests_num << " AIO requests...";
    int64_t left_requests_num = submit_requests_num;
    int retry_count = 0;
    {
      std::lock_guard<std::mutex> lock(submitted_io_requests_mutex_);
      for (int64_t i = 0; i < submit_requests_num; ++i) {
        user_data_vec[i]->submit_timestamp =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
        submitted_io_requests_.insert(user_data_vec[i]);
      }
    }
    do {
      int submitted = aio_ops_->io_submit(
          aio_ctx_, left_requests_num,
          iocbs.get() + (submit_requests_num - left_requests_num));
      if (submitted < 0) {
        LOG(ERROR) << "io_submit failed: " << strerror(-submitted)
                   << ", current submitted requests: "
                   << GetProcessingIORequest() << ", retrying...";
        usleep(std::stoi(VLLMKVCacheEnv::AIORetryWaitMicroseconds()));
        retry_count++;
        if (retry_count >= retry_times_) {
          LOG(ERROR) << "Failed to submit AIO requests after " << retry_count
                     << " retries.";
          // set failed promises
          std::lock_guard<std::mutex> lock(submitted_io_requests_mutex_);
          for (int64_t i = 0; i < left_requests_num; ++i) {
            auto user_data = static_cast<AIOUserData*>(
                iocbs[submit_requests_num - left_requests_num + i]->data);
            try {
              bool promise_set = user_data->promise_setted.exchange(true);
              if (!promise_set) {
                user_data->promise.set_value(Status::IOError(
                    "Failed to submit AIO request after retries"));
              } else {
                LOG(WARNING)
                    << "Promise already set for AIO operation, skipping.";
              }
            } catch (const std::future_error& e) {
              LOG(WARNING) << "Failed to set promise value: " << e.what()
                           << " maybe set by timeout tracker.";
            }
            submitted_io_requests_.erase(user_data);
            ReleaseAIOUserData(user_data);
          }
          break;
        }
      } else {
        submitted_requests_.fetch_add(submitted);
        left_requests_num -= submitted;
      }
    } while (left_requests_num > 0);
  }
  LOG(INFO) << "Submit thread stopped.";
}

AIOContext::~AIOContext() {
  stop_.store(true);
  for (size_t i = 0; i < concurrency_; ++i) {
    queue_cv_[i]->notify_all();
  }
  thread_pool_.reset();  // Reset the thread pool to stop all threads
  aio_ops_->io_destroy(aio_ctx_);
  LOG(INFO) << "AIOContext destroyed.";
}

AIOUserData* AIOContext::CreateAIOUserData(size_t expect_size,
                                           std::promise<Status> promise) {
  auto user_data = new AIOUserData();
  user_data->expect_size = expect_size;
  user_data->promise = std::move(promise);
  memset(&user_data->cb, 0, sizeof(user_data->cb));
  return user_data;
}

void AIOContext::ReleaseAIOUserData(AIOUserData* user_data) {
  if (user_data) {
    delete user_data;
  }
}

IOAdaptorFactory GetAIOAdaptorFactory() {
  LOG(INFO) << "Using AIO adaptor";
  return [](const std::string& path)
             -> std::shared_ptr<vllm_kv_cache::io::IIOAdaptor> {
    return std::make_shared<AIOAdaptor>(path);
  };
}

std::future<Status> AIOContext::SubmitRead(int fd, void* data, size_t size,
                                           size_t offset) {
  std::shared_ptr<std::promise<Status>> status_promise =
      std::make_shared<std::promise<Status>>();
  auto status_future = status_promise->get_future();
  if (fd < 0) {
    status_promise->set_value(Status::IOError("Invalid file descriptor"));
    return status_future;
  }

  int64_t queue_index = read_counter_.fetch_add(1) % concurrency_;
  auto user_data = CreateAIOUserData(size, std::move(*status_promise));
  aio_ops_->io_prep_pread(&user_data->cb, fd, data, size, offset);
  user_data->cb.data = user_data;  // Set the user data pointer for completion
  user_data->timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();

  std::vector<AIOUserData*> user_data_vec;
  user_data_vec.push_back(user_data);

  PushRequest(user_data_vec, queue_index);
  return status_future;
}

std::future<Status> AIOContext::SubmitRead(int fd, std::shared_ptr<void> data,
                                           size_t size, size_t offset) {
  std::shared_ptr<std::promise<Status>> status_promise =
      std::make_shared<std::promise<Status>>();
  auto status_future = status_promise->get_future();
  if (fd < 0) {
    status_promise->set_value(Status::IOError("Invalid file descriptor"));
    return status_future;
  }

  int64_t queue_index = read_counter_.fetch_add(1) % concurrency_;
  auto user_data = CreateAIOUserData(size, std::move(*status_promise));
  aio_ops_->io_prep_pread(&user_data->cb, fd, data.get(), size, offset);
  user_data->cb.data = user_data;  // Set the user data pointer for completion
  user_data->data_ptr = data;      // Store the shared pointer
  user_data->timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();

  std::vector<AIOUserData*> user_data_vec;
  user_data_vec.push_back(user_data);

  PushRequest(user_data_vec, queue_index);
  return status_future;
}

std::future<Status> AIOContext::SubmitWrite(int fd, void* data, size_t size,
                                            size_t offset) {
  std::shared_ptr<std::promise<Status>> status_promise =
      std::make_shared<std::promise<Status>>();
  auto status_future = status_promise->get_future();
  if (fd < 0) {
    status_promise->set_value(Status::IOError("Invalid file descriptor"));
    return status_future;
  }

  int64_t queue_index = write_counter_.fetch_add(1) % concurrency_;
  auto user_data = CreateAIOUserData(size, std::move(*status_promise));
  aio_ops_->io_prep_pwrite(&user_data->cb, fd, data, size, offset);
  user_data->cb.data = user_data;  // Set the user data pointer for completion
  user_data->timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();

  std::vector<AIOUserData*> user_data_vec;
  user_data_vec.push_back(user_data);

  PushRequest(user_data_vec, queue_index);
  return status_future;
}

std::future<Status> AIOContext::SubmitWrite(int fd, std::shared_ptr<void> data,
                                            size_t size, size_t offset) {
  std::shared_ptr<std::promise<Status>> status_promise =
      std::make_shared<std::promise<Status>>();
  auto status_future = status_promise->get_future();
  if (fd < 0) {
    status_promise->set_value(Status::IOError("Invalid file descriptor"));
    return status_future;
  }

  int64_t queue_index = write_counter_.fetch_add(1) % concurrency_;
  auto user_data = CreateAIOUserData(size, std::move(*status_promise));
  aio_ops_->io_prep_pwrite(&user_data->cb, fd, data.get(), size, offset);
  user_data->cb.data = user_data;  // Set the user data pointer for completion
  user_data->data_ptr = data;      // Store the shared pointer
  user_data->timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();

  std::vector<AIOUserData*> user_data_vec;
  user_data_vec.push_back(user_data);

  PushRequest(user_data_vec, queue_index);
  return status_future;
}

Status AIOContext::BatchSubmitRead(int fd, std::vector<void*>& data_vec,
                                   std::vector<size_t>& size_vec,
                                   std::vector<size_t>& offset_vec,
                                   std::vector<std::future<Status>>& results) {
  if (fd < 0) {
    LOG(ERROR) << "Invalid file descriptor: " << fd;
    return Status::IOError("Invalid file descriptor");
  }

  uint64_t queue_index = read_counter_.fetch_add(1) % concurrency_;

  std::vector<std::shared_ptr<std::promise<Status>>> status_promise_vec;
  for (size_t i = 0; i < data_vec.size(); ++i) {
    status_promise_vec.push_back(std::make_shared<std::promise<Status>>());
    results.push_back(status_promise_vec.back()->get_future());
  }

  std::vector<AIOUserData*> user_data_vec;
  for (size_t i = 0; i < data_vec.size(); ++i) {
    auto user_data =
        this->CreateAIOUserData(size_vec[i], std::move(*status_promise_vec[i]));
    aio_ops_->io_prep_pread(&user_data->cb, fd, data_vec[i], size_vec[i],
                            offset_vec[i]);
    user_data->cb.data = user_data;  // Set the user data pointer for completion
    user_data_vec.push_back(user_data);
  }

  PushRequest(user_data_vec, queue_index);
  return Status::OK();
}

Status AIOContext::BatchSubmitRead(int fd,
                                   std::vector<std::shared_ptr<void>>& data_vec,
                                   std::vector<size_t>& size_vec,
                                   std::vector<size_t>& offset_vec,
                                   std::vector<std::future<Status>>& results) {
  if (fd < 0) {
    LOG(ERROR) << "Invalid file descriptor: " << fd;
    return Status::IOError("Invalid file descriptor");
  }

  uint64_t queue_index = read_counter_.fetch_add(1) % concurrency_;

  std::vector<std::shared_ptr<std::promise<Status>>> status_promise_vec;
  for (size_t i = 0; i < data_vec.size(); ++i) {
    status_promise_vec.push_back(std::make_shared<std::promise<Status>>());
    results.push_back(status_promise_vec.back()->get_future());
  }

  std::vector<AIOUserData*> user_data_vec;
  for (size_t i = 0; i < data_vec.size(); ++i) {
    auto user_data =
        this->CreateAIOUserData(size_vec[i], std::move(*status_promise_vec[i]));
    aio_ops_->io_prep_pread(&user_data->cb, fd, data_vec[i].get(), size_vec[i],
                            offset_vec[i]);
    user_data->cb.data = user_data;  // Set the user data pointer for completion
    user_data->data_ptr = data_vec[i];  // Store the shared pointer
    user_data_vec.push_back(user_data);
  }

  PushRequest(user_data_vec, queue_index);
  return Status::OK();
}

Status AIOContext::BatchSubmitWrite(int fd, std::vector<void*>& data_vec,
                                    std::vector<size_t>& size_vec,
                                    std::vector<size_t>& offset_vec,
                                    std::vector<std::future<Status>>& results) {
  if (fd < 0) {
    LOG(ERROR) << "Invalid file descriptor: " << fd;
    return Status::Invalid("Invalid file descriptor");
  }

  uint64_t queue_index = write_counter_.fetch_add(1) % concurrency_;

  std::vector<std::shared_ptr<std::promise<Status>>> status_promise_vec;
  for (size_t i = 0; i < data_vec.size(); ++i) {
    status_promise_vec.push_back(std::make_shared<std::promise<Status>>());
    results.push_back(status_promise_vec.back()->get_future());
  }

  std::vector<AIOUserData*> user_data_vec;
  for (size_t i = 0; i < data_vec.size(); ++i) {
    auto user_data =
        CreateAIOUserData(size_vec[i], std::move(*status_promise_vec[i]));
    aio_ops_->io_prep_pwrite(&user_data->cb, fd, data_vec[i], size_vec[i],
                             offset_vec[i]);
    user_data->cb.data = user_data;  // Set the user data pointer for completion
    user_data_vec.push_back(user_data);
  }

  PushRequest(user_data_vec, queue_index);
  return Status::OK();
}

Status AIOContext::BatchSubmitWrite(
    int fd, std::vector<std::shared_ptr<void>>& data_vec,
    std::vector<size_t>& size_vec, std::vector<size_t>& offset_vec,
    std::vector<std::future<Status>>& results) {
  if (fd < 0) {
    LOG(ERROR) << "Invalid file descriptor: " << fd;
    return Status::Invalid("Invalid file descriptor");
  }

  uint64_t queue_index = write_counter_.fetch_add(1) % concurrency_;

  std::vector<std::shared_ptr<std::promise<Status>>> status_promise_vec;
  for (size_t i = 0; i < data_vec.size(); ++i) {
    status_promise_vec.push_back(std::make_shared<std::promise<Status>>());
    results.push_back(status_promise_vec.back()->get_future());
  }

  std::vector<AIOUserData*> user_data_vec;
  for (size_t i = 0; i < data_vec.size(); ++i) {
    auto user_data =
        CreateAIOUserData(size_vec[i], std::move(*status_promise_vec[i]));
    aio_ops_->io_prep_pwrite(&user_data->cb, fd, data_vec[i].get(), size_vec[i],
                             offset_vec[i]);
    user_data->cb.data = user_data;  // Set the user data pointer for completion
    user_data->data_ptr = data_vec[i];  // Store the shared pointer
    user_data_vec.push_back(user_data);
  }

  PushRequest(user_data_vec, queue_index);
  return Status::OK();
}

void AIOContext::PullRequest(std::vector<AIOUserData*>& user_data,
                             uint64_t queue_index) {
  std::unique_lock<std::mutex> ulock(queue_mutex_[queue_index]);
  if (io_queue_[queue_index].empty()) {
    // 等待在 condition virable 上
    // wait 直到有新的请求被加入
    this->queue_cv_[queue_index]->wait(ulock, [this, queue_index] {
      return !this->io_queue_[queue_index].empty() || this->stop_.load();
    });
    if (this->stop_.load()) {
      return;
    }
  }

  while (!this->io_queue_[queue_index].empty() &&
         user_data.size() < static_cast<uint64_t>(max_push_events_)) {
    user_data.push_back(this->io_queue_[queue_index].front());
    this->io_queue_[queue_index].pop_front();
    user_data.back()->dequeue_timestamp =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  }

  static uint64_t start =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  if (!user_data.empty()) {
    uint64_t end = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
    if (end - start > 100) {
      LOG(INFO) << "Pulled " << user_data.size()
                << " requests from queue index " << queue_index
                << " pendding requests: "
                << this->io_queue_[queue_index].size();
      start = end;
    }
  }
}

void AIOContext::PushRequest(std::vector<AIOUserData*>& user_data,
                             uint64_t queue_index) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_[queue_index]);
    for (auto data : user_data) {
      data->enqueue_timestamp =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
      this->io_queue_[queue_index].push_back(data);
    }
  }
  // 唤醒等待的线程
  this->queue_cv_[queue_index]->notify_one();
}

Status AIOAdaptor::Open(std::string mode, bool direct_io) {
  int flags = 0;
  if (mode == "r") {
    flags = O_RDONLY;
  } else if (mode == "w") {
    flags = O_WRONLY | O_CREAT | O_TRUNC;
  } else {
    return Status::Invalid("Invalid mode: " + mode);
  }

  if (direct_io) {
    flags |= O_DIRECT;
  }
  direct_io_ = direct_io;

  fd_ = open(location_.c_str(), flags, 0666);
  if (fd_ == -1) {
    return Status::IOError("Failed to open file: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }
  return Status::OK();
}

Status AIOAdaptor::Read(void* data, size_t size) {
  if (fd_ == -1) {
    return Status::IOError("File not opened: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }

  if (direct_io_ && size % std::stoi(VLLMKVCacheEnv::GetDirectIOAlign()) != 0) {
    return Status::IOError(
        "Direct IO requires size to be a multiple of 512 bytes, but got: " +
        std::to_string(size));
  }

  auto future_status =
      aio_context_->SubmitRead(this->fd_, data, size, this->read_pos);
  if (future_status.valid()) {
    Status status = future_status.get();
    if (status.ok()) {
      read_pos += size;  // Update read position
    }
    return status;
  } else {
    return Status::IOError("Failed to submit read operation");
  }
}

Status AIOAdaptor::Write(void* data, size_t size) {
  if (fd_ == -1) {
    return Status::IOError("File not opened: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }

  auto future_status =
      aio_context_->SubmitWrite(this->fd_, data, size, this->write_pos);
  if (future_status.valid()) {
    Status status = future_status.get();
    if (status.ok()) {
      write_pos += size;  // Update write position
    }
    return status;
  } else {
    return Status::IOError("Failed to submit write operation");
  }
}

Status AIOAdaptor::Close() {
  if (fd_ == -1) {
    return Status::OK();  // No error, just return OK
  }

  if (direct_io_ && write_pos) {
    int align_size = std::stoi(VLLMKVCacheEnv::GetDirectIOAlign());
    if (write_pos % align_size != 0) {
      LOG(WARNING) << "Direct IO requires write position to be a multiple of "
                   << align_size << ", padding with zeros.";
      // Pad with zeros to align the write position
      size_t padding_size = align_size - (write_pos % align_size);
      char* padding_data = new char[padding_size];
      memset(padding_data, 0, padding_size);
      Status status = Write(padding_data, padding_size);
      delete[] padding_data;
      if (!status.ok()) {
        LOG(ERROR) << "Failed to write padding data for direct IO: "
                   << status.ToString();
        return status;
      }
    }
  }

  if (fsync(fd_) < 0) {
    return Status::IOError("Failed to sync file: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }

  if (close(fd_) < 0) {
    return Status::IOError("Failed to close file: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }
  fd_ = -1;
  return Status::OK();
}

Status AIOAdaptor::FileTruncate(size_t size) {
  if (fd_ == -1) {
    return Status::IOError("File not opened: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }

  if (ftruncate(fd_, size) < 0) {
    return Status::IOError("Failed to truncate file: " + location_ +
                           " to size: " + std::to_string(size) +
                           " error:" + std::string(strerror(errno)));
  }
  return Status::OK();
}

std::future<Status> AIOAdaptor::AsyncRead(void* data, size_t size,
                                          size_t offset) {
  if (fd_ == -1) {
    return std::async(std::launch::async, [this]() {
      return Status::IOError("File not opened: " + location_ +
                             " error:" + std::string(strerror(errno)));
    });
  }

  return aio_context_->SubmitRead(this->fd_, data, size, offset);
}

std::future<Status> AIOAdaptor::AsyncRead(std::shared_ptr<void> data,
                                          size_t size, size_t offset) {
  if (fd_ == -1) {
    return std::async(std::launch::async, [this]() {
      return Status::IOError("File not opened: " + location_ +
                             " error:" + std::string(strerror(errno)));
    });
  }

  return aio_context_->SubmitRead(this->fd_, data, size, offset);
}

std::future<Status> AIOAdaptor::AsyncWrite(void* data, size_t size,
                                           size_t offset) {
  if (fd_ == -1) {
    return std::async(std::launch::async, [this]() {
      return Status::IOError("File not opened: " + location_ +
                             " error:" + std::string(strerror(errno)));
    });
  }

  return aio_context_->SubmitWrite(this->fd_, data, size, offset);
}

std::future<Status> AIOAdaptor::AsyncWrite(std::shared_ptr<void> data,
                                           size_t size, size_t offset) {
  if (fd_ == -1) {
    return std::async(std::launch::async, [this]() {
      return Status::IOError("File not opened: " + location_ +
                             " error:" + std::string(strerror(errno)));
    });
  }

  return aio_context_->SubmitWrite(this->fd_, data, size, offset);
}

Status AIOAdaptor::BatchAsyncRead(std::vector<void*>& data_vec,
                                  std::vector<size_t>& size_vec,
                                  std::vector<size_t>& offset_vec,
                                  std::vector<std::future<Status>>& results) {
  if (fd_ == -1) {
    return Status::IOError("File not opened: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }

  return aio_context_->BatchSubmitRead(this->fd_, data_vec, size_vec,
                                       offset_vec, results);
}

Status AIOAdaptor::BatchAsyncRead(std::vector<std::shared_ptr<void>>& data_vec,
                                  std::vector<size_t>& size_vec,
                                  std::vector<size_t>& offset_vec,
                                  std::vector<std::future<Status>>& results) {
  if (fd_ == -1) {
    return Status::IOError("File not opened: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }

  return aio_context_->BatchSubmitRead(this->fd_, data_vec, size_vec,
                                       offset_vec, results);
}

Status AIOAdaptor::BatchAsyncWrite(std::vector<void*>& data_vec,
                                   std::vector<size_t>& size_vec,
                                   std::vector<size_t>& offset_vec,
                                   std::vector<std::future<Status>>& results) {
  if (fd_ == -1) {
    LOG(ERROR) << "File not opened: " << location_
               << ", error: " << std::string(strerror(errno));
    return Status::IOError("File not opened: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }

  return aio_context_->BatchSubmitWrite(this->fd_, data_vec, size_vec,
                                        offset_vec, results);
}

Status AIOAdaptor::BatchAsyncWrite(std::vector<std::shared_ptr<void>>& data_vec,
                                   std::vector<size_t>& size_vec,
                                   std::vector<size_t>& offset_vec,
                                   std::vector<std::future<Status>>& results) {
  if (fd_ == -1) {
    LOG(ERROR) << "File not opened: " << location_
               << ", error: " << std::string(strerror(errno));
    return Status::IOError("File not opened: " + location_ +
                           " error:" + std::string(strerror(errno)));
  }

  return aio_context_->BatchSubmitWrite(this->fd_, data_vec, size_vec,
                                        offset_vec, results);
}

Status AIOAdaptor::GetFileSize(size_t& size) {
  std::filesystem::path file_path(location_);
  if (!std::filesystem::exists(file_path)) {
    return Status::IOError("File does not exist: " + location_);
  }

  std::error_code ec;
  size = std::filesystem::file_size(file_path, ec);
  if (ec) {
    LOG(ERROR) << "Error getting file size for '" << file_path.string()
               << "': " << ec.message();
    return Status::IOError("Failed to get file size: " + file_path.string() +
                           ", error: " + ec.message());
  }

  return Status::OK();
}

}  // namespace io

}  // namespace vllm_kv_cache

}  // namespace vineyard
