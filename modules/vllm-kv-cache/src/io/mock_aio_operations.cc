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

#include "vllm-kv-cache/src/io/mock_aio_operations.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <memory>

#include "common/util/logging.h"
#include "vllm-kv-cache/src/io/error_injection.h"

namespace vineyard {

namespace vllm_kv_cache {

namespace io {

MockAIOOperations::MockAIOOperations() : stop_worker_(false) {
  // Start worker thread
  worker_thread_ = std::thread(&MockAIOOperations::ProcessIORequests, this);
}

MockAIOOperations::~MockAIOOperations() {
  // Stop worker thread
  {
    std::lock_guard<std::mutex> lock(pending_ios_mutex_);
    stop_worker_ = true;
    pending_ios_cv_.notify_all();
  }

  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }

  // Destroy all contexts
  std::lock_guard<std::mutex> lock(contexts_mutex_);
  for (auto& pair : contexts_) {
    pair.second->destroyed = true;
    pair.second->cv.notify_all();
  }
  contexts_.clear();
}

int MockAIOOperations::io_setup(int maxevents, io_context_t* ctx_idp) {
  LOG(INFO) << "mock io_setup";
  if (global_mock_aio_operation_io_setup_error) {
    LOG(ERROR) << "in mock aio operations: io_setup failed due to global "
                  "injection, error code: "
               << global_mock_aio_operation_io_setup_error_code;
    return global_mock_aio_operation_io_setup_error_code;
  }

  // Create a new context
  auto context = std::make_shared<AIOContext>(maxevents);

  // Generate a unique context ID (using pointer address)
  io_context_t ctx_id = reinterpret_cast<io_context_t>(new char);

  // Store context
  {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    contexts_[ctx_id] = context;
  }

  *ctx_idp = ctx_id;
  return 0;
}

int MockAIOOperations::io_submit(io_context_t ctx_id, int64_t nr,
                                 struct iocb* ios[]) {
  if (global_mock_aio_operation_io_submit_timeout) {
    LOG(INFO) << "io_submit timeout injected for "
              << global_mock_aio_operation_io_submit_timeout_ms << " ms";
    std::this_thread::sleep_for(std::chrono::milliseconds(
        global_mock_aio_operation_io_submit_timeout_ms));
  }

  if (global_mock_aio_operation_io_submit_error) {
    LOG(ERROR) << "in mock aio operations: io_submit failed due to global "
                  "injection, error code: "
               << global_mock_aio_operation_io_submit_error_code;
    return global_mock_aio_operation_io_submit_error_code;
  }

  // Find context
  std::shared_ptr<AIOContext> context;
  {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    auto it = contexts_.find(ctx_id);
    if (it == contexts_.end()) {
      LOG(ERROR) << "in mock aio operations: io_submit failed: context not "
                    "found for ctx_id="
                 << ctx_id;
      return -EINVAL;
    }
    context = it->second;
  }

  if (context->destroyed) {
    LOG(ERROR) << "in mock aio operations: io_submit failed: context is "
                  "destroyed for ctx_id="
               << ctx_id;
    return -EBADF;
  }

  // Determine how many requests to process
  int64_t processed =
      (global_mock_aio_operation_io_submit_part_processed == false)
          ? nr
          : std::min(nr, global_mock_aio_operation_io_submit_max_processed);

  // Add I/O requests to pending queue
  {
    std::lock_guard<std::mutex> lock(pending_ios_mutex_);
    for (int64_t i = 0; i < processed; i++) {
      pending_ios_.push({ctx_id, ios[i]});
    }
    pending_ios_cv_.notify_all();
  }

  if (global_mock_aio_operation_io_submit_part_processed) {
    LOG(INFO) << "io_submit part processed injected for max processed: "
              << global_mock_aio_operation_io_submit_max_processed
              << " requests, real processed: " << processed << " requests";
  }

  return processed;
}

int MockAIOOperations::io_getevents(io_context_t ctx_id, int64_t min_nr,
                                    int64_t nr, struct io_event* events,
                                    struct timespec* timeout) {
  if (global_mock_aio_operation_io_getevents_no_events) {
    LOG(INFO) << "io_getevents returning 0 events due to global injection";
    return 0;
  }

  // Find context
  std::shared_ptr<AIOContext> context;
  {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    auto it = contexts_.find(ctx_id);
    if (it == contexts_.end()) {
      LOG(ERROR) << "in mock aio operations: io_getevents failed: context not "
                    "found for ctx_id="
                 << ctx_id;
      return -EINVAL;
    }
    context = it->second;
  }

  if (context->destroyed) {
    LOG(ERROR) << "in mock aio operations: io_getevents failed: context is "
                  "destroyed for ctx_id="
               << ctx_id;
    return -EBADF;
  }

  // Convert timespec to chrono duration
  std::chrono::steady_clock::time_point deadline;
  bool has_timeout = false;
  if (timeout) {
    deadline = std::chrono::steady_clock::now() +
               std::chrono::seconds(timeout->tv_sec) +
               std::chrono::nanoseconds(timeout->tv_nsec);
    has_timeout = true;
  }

  int64_t num_events = 0;
  std::unique_lock<std::mutex> lock(context->mutex);

  // Wait for events if needed
  if (min_nr > 0 &&
      context->completed_events.size() < static_cast<size_t>(min_nr)) {
    if (has_timeout) {
      context->cv.wait_until(lock, deadline, [context, min_nr]() {
        return context->completed_events.size() >=
                   static_cast<size_t>(min_nr) ||
               context->destroyed;
      });
    } else {
      context->cv.wait(lock, [context, min_nr]() {
        return context->completed_events.size() >=
                   static_cast<size_t>(min_nr) ||
               context->destroyed;
      });
    }
  }

  if (context->destroyed) {
    return -EBADF;
  }

  // Check for timeout
  if (has_timeout && std::chrono::steady_clock::now() > deadline) {
    if (global_mock_aio_operation_io_getevents_timeout) {
      // LOG(ERROR) << "in mock aio operations: io_getevents timed out, error
      // code: " << -EAGAIN;
      return -EAGAIN;
    }
  }

  // Collect events
  while (num_events < nr && !context->completed_events.empty()) {
    events[num_events++] = context->completed_events.front();
    context->completed_events.pop();
  }

  if (global_mock_aio_operation_io_getevents_timeout && num_events > 0) {
    // if the first event is a short read(meta), we don't set timeout
    if (num_events == 1 && events[0].res <= 4096) {
      // Do nothing
    } else {
      LOG(INFO) << "io_getevents timeout injected for "
                << global_mock_aio_operation_io_getevents_timeout_ms << " ms";
      std::this_thread::sleep_for(std::chrono::milliseconds(
          global_mock_aio_operation_io_getevents_timeout_ms));
    }
  }

  if (global_mock_aio_operation_io_getevents_error && num_events > 0) {
    // if the first event is a short read(meta), we don't set timeout
    if (num_events == 1 && events[0].res <= 4096) {
      // Do nothing
    } else {
      LOG(ERROR) << "in mock aio operations: io_getevents failed due to global "
                    "injection, error code: "
                 << global_mock_aio_operation_io_getevents_error_code;
      return global_mock_aio_operation_io_getevents_error_code;
    }
  }

  return num_events;
}

int MockAIOOperations::io_destroy(io_context_t ctx_id) {
  std::shared_ptr<AIOContext> context;
  {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    auto it = contexts_.find(ctx_id);
    if (it == contexts_.end()) {
      LOG(ERROR) << "in mock aio operations: io_destroy failed: context not "
                    "found for ctx_id="
                 << ctx_id;
      return -EINVAL;
    }
    context = it->second;
    contexts_.erase(it);
  }

  // Mark context as destroyed and notify any waiting threads
  context->destroyed = true;
  context->cv.notify_all();

  // Clean up context ID
  delete[] reinterpret_cast<char*>(ctx_id);

  return 0;
}

// Mock functions for io_prep_pread and io_prep_pwrite
void MockAIOOperations::io_prep_pread(struct iocb* iocb, int fd, void* buf,
                                      size_t count, int64_t offset) {
  std::memset(iocb, 0, sizeof(*iocb));
  iocb->aio_fildes = fd;
  iocb->aio_lio_opcode = IO_CMD_PREAD;
  iocb->aio_reqprio = 0;
  // Use the union member 'c' which has buf, nbytes, and offset
  iocb->u.c.buf = buf;
  iocb->u.c.nbytes = count;
  iocb->u.c.offset = offset;
}

void MockAIOOperations::io_prep_pwrite(struct iocb* iocb, int fd, void* buf,
                                       size_t count, int64_t offset) {
  std::memset(iocb, 0, sizeof(*iocb));
  iocb->aio_fildes = fd;
  iocb->aio_lio_opcode = IO_CMD_PWRITE;
  iocb->aio_reqprio = 0;
  // Use the union member 'c' which has buf, nbytes, and offset
  iocb->u.c.buf = buf;
  iocb->u.c.nbytes = count;
  iocb->u.c.offset = offset;
}

void MockAIOOperations::ProcessIORequests() {
  while (!stop_worker_) {
    std::vector<PendingIO> io_requests;

    // Wait for pending I/O requests
    {
      std::unique_lock<std::mutex> lock(pending_ios_mutex_);
      pending_ios_cv_.wait(
          lock, [this]() { return !pending_ios_.empty() || stop_worker_; });

      if (stop_worker_ && pending_ios_.empty()) {
        break;
      }

      // Dequeue all pending I/O requests
      while (!pending_ios_.empty()) {
        io_requests.push_back(pending_ios_.front());
        pending_ios_.pop();
      }
    }

    // Process all I/O requests concurrently
    std::vector<std::thread> workers;
    for (const auto& io_request : io_requests) {
      workers.emplace_back([this, io_request]() {
        SimulateIOOperation(io_request.ctx_id, io_request.iocb_ptr);
      });
    }

    // Wait for all workers to complete
    for (auto& worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }
}

void MockAIOOperations::SimulateIOOperation(io_context_t ctx_id,
                                            struct iocb* iocb) {
  // For simplicity, we'll just complete the operation immediately
  // In a more realistic implementation, you might want to simulate actual I/O
  // delays
  ssize_t result = 0;

  switch (iocb->aio_lio_opcode) {
  case IO_CMD_PREAD: {
    // Simulate reading from a file
    result = ReadFromFile(iocb->aio_fildes, iocb->u.c.buf,
                          static_cast<size_t>(iocb->u.c.nbytes),
                          static_cast<off_t>(iocb->u.c.offset));
    break;
  }
  case IO_CMD_PWRITE: {
    // Simulate writing to a file
    result = WriteToFile(iocb->aio_fildes, iocb->u.c.buf,
                         static_cast<size_t>(iocb->u.c.nbytes),
                         static_cast<off_t>(iocb->u.c.offset));
    break;
  }
  default:
    LOG(ERROR)
        << "in mock aio operations: SimulateIOOperation failed: unknown opcode="
        << iocb->aio_lio_opcode;
    result = -EINVAL;
    break;
  }

  CompleteIOOperation(ctx_id, iocb, result);
}

void MockAIOOperations::CompleteIOOperation(io_context_t ctx_id,
                                            struct iocb* iocb, ssize_t result) {
  // Find context
  std::shared_ptr<AIOContext> context;
  {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    auto it = contexts_.find(ctx_id);
    if (it == contexts_.end()) {
      LOG(ERROR) << "in mock aio operations: CompleteIOOperation failed: "
                    "context not found for ctx_id="
                 << ctx_id;
      return;
    }
    context = it->second;
  }

  if (context->destroyed) {
    LOG(ERROR) << "in mock aio operations: CompleteIOOperation failed: context "
                  "is destroyed for ctx_id="
               << ctx_id;
    return;
  }

  // Create io_event
  struct io_event event;
  std::memset(&event, 0, sizeof(event));
  event.data = reinterpret_cast<void*>(iocb);
  event.obj = iocb;
  event.res = result;
  event.res2 = 0;

  // Add event to completed events queue
  {
    std::lock_guard<std::mutex> lock(context->mutex);
    context->completed_events.push(event);
  }

  // Notify waiting threads
  context->cv.notify_all();
}

ssize_t MockAIOOperations::ReadFromFile(int fd, void* buf, size_t count,
                                        off_t offset) {
  LOG(INFO) << "in mock aio operations: ReadFromFile called for fd=" << fd
            << ", count=" << count << ", offset=" << offset;

  // Inject read error if enabled
  if (global_mock_aio_operation_io_read_error) {
    LOG(ERROR) << "in mock aio operations: ReadFromFile failed due to global "
                  "injection";
    return -EIO;
  }

  // Find file data
  std::shared_ptr<FileData> file_data;
  {
    std::lock_guard<std::mutex> lock(files_mutex_);
    auto it = files_.find(fd);
    if (it == files_.end()) {
      // File not found, return error
      LOG(ERROR) << "in mock aio operations: ReadFromFile failed: file not "
                    "found for fd="
                 << fd;
      return -EBADF;
    }
    file_data = it->second;
  }

  size_t to_read = 0;
  {
    // Lock file data for reading
    std::lock_guard<std::mutex> lock(file_data->mutex);

    // Check if offset is beyond file size
    if (static_cast<size_t>(offset) >= file_data->data.size()) {
      return 0;  // EOF
    }

    // Calculate number of bytes to read
    size_t available = file_data->data.size() - static_cast<size_t>(offset);
    to_read = std::min(count, available);

    // Copy data to buffer
    std::memcpy(buf, file_data->data.data() + offset, to_read);
  }

  // Inject read timeout if enabled
  if (global_mock_aio_operation_io_read_timeout && count > 4096) {
    LOG(INFO) << "ReadFromFile timeout injected for "
              << global_mock_aio_operation_io_timeout_ms << " ms";
    std::this_thread::sleep_for(
        std::chrono::milliseconds(global_mock_aio_operation_io_timeout_ms));
  }

  return static_cast<ssize_t>(to_read);
}

ssize_t MockAIOOperations::WriteToFile(int fd, const void* buf, size_t count,
                                       off_t offset) {
  // LOG(INFO) << "in mock aio operations: WriteToFile called for fd=" << fd
  // << ", count=" << count << ", offset=" << offset;

  // Inject write error if enabled
  if (global_mock_aio_operation_io_write_error) {
    LOG(ERROR)
        << "in mock aio operations: WriteToFile failed due to global injection";
    return -EIO;
  }

  // Inject write timeout if enabled
  if (global_mock_aio_operation_io_write_timeout) {
    LOG(INFO) << "WriteToFile timeout injected for "
              << global_mock_aio_operation_io_timeout_ms << " ms";
    std::this_thread::sleep_for(
        std::chrono::milliseconds(global_mock_aio_operation_io_timeout_ms));
  }

  // Find or create file data
  std::shared_ptr<FileData> file_data;
  {
    std::lock_guard<std::mutex> lock(files_mutex_);
    auto it = files_.find(fd);
    if (it == files_.end()) {
      // Create new file data
      file_data = std::make_shared<FileData>();
      files_[fd] = file_data;
    } else {
      file_data = it->second;
    }
  }

  // Lock file data for writing
  std::lock_guard<std::mutex> lock(file_data->mutex);

  // Ensure file data is large enough
  size_t required_size = static_cast<size_t>(offset) + count;
  if (file_data->data.size() < required_size) {
    file_data->data.resize(required_size);
  }

  // Copy data from buffer
  std::memcpy(file_data->data.data() + offset, buf, count);

  return static_cast<ssize_t>(count);
}

}  // namespace io

}  // namespace vllm_kv_cache

}  // namespace vineyard
