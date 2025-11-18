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

#include "vllm-kv-cache/src/io/error_injection.h"
#include <errno.h>
#include <stdint.h>

namespace vineyard {

namespace vllm_kv_cache {

namespace io {

// Global error injection flags for MockAIOOperations
bool global_mock_aio_operation_io_setup_error = false;
bool global_mock_aio_operation_io_submit_timeout = false;
bool global_mock_aio_operation_io_submit_error = false;
bool global_mock_aio_operation_io_submit_part_processed = false;
int64_t global_mock_aio_operation_io_submit_max_processed = 5;
bool global_mock_aio_operation_io_getevents_timeout = false;
bool global_mock_aio_operation_io_getevents_error = false;
bool global_mock_aio_operation_io_getevents_no_events = false;
uint64_t global_mock_aio_operation_io_submit_timeout_ms = 1000;
uint64_t global_mock_aio_operation_io_getevents_timeout_ms = 1000;
int global_mock_aio_operation_io_setup_error_code = -EAGAIN;
int global_mock_aio_operation_io_submit_error_code = -EAGAIN;
int global_mock_aio_operation_io_getevents_error_code = -EIO;

bool global_mock_aio_operation_io_read_error = false;
bool global_mock_aio_operation_io_write_error = false;
bool global_mock_aio_operation_io_read_timeout = false;
bool global_mock_aio_operation_io_write_timeout = false;
uint64_t global_mock_aio_operation_io_timeout_ms = 1000;

// Global error injection flags for MockIOAdaptor
bool global_mock_io_read_error = false;
bool global_mock_io_write_error = false;
bool global_mock_io_read_timeout = false;
bool global_mock_io_write_timeout = false;
bool global_mock_io_batch_read_error = false;
bool global_mock_io_batch_write_error = false;
bool global_mock_io_batch_read_timeout = false;
bool global_mock_io_batch_write_timeout = false;
uint64_t global_mock_io_timeout_ms = 1000;

// Functions to set global error injection flags for MockAIOOperations
void SetGlobalMockAIOOperationSetupError(bool error) {
  global_mock_aio_operation_io_setup_error = error;
}

void SetGlobalMockAIOOperationSubmitTimeout(bool timeout, uint64_t timeout_ms) {
  global_mock_aio_operation_io_submit_timeout = timeout;
  global_mock_aio_operation_io_submit_timeout_ms = timeout_ms;
}

void SetGlobalMockAIOOperationSubmitError(bool error) {
  global_mock_aio_operation_io_submit_error = error;
}

void SetGlobalMockAIOOperationSubmitMaxProcessedPerCall(
    bool is_part_processed, uint64_t max_processed) {
  global_mock_aio_operation_io_submit_part_processed = is_part_processed;
  global_mock_aio_operation_io_submit_max_processed = max_processed;
}

void SetGlobalMockAIOOperationGetEventsTimeout(bool timeout,
                                               uint64_t timeout_ms) {
  global_mock_aio_operation_io_getevents_timeout = timeout;
  global_mock_aio_operation_io_getevents_timeout_ms = timeout_ms;
}

void SetGlobalMockAIOOperationGetEventsError(bool error) {
  global_mock_aio_operation_io_getevents_error = error;
}

void SetGlobalMockAIOOperationGetEventsNoEvents(bool no_events) {
  global_mock_aio_operation_io_getevents_no_events = no_events;
}

void SetGlobalMockAIOOperationReadError(bool error) {
  global_mock_aio_operation_io_read_error = error;
}

void SetGlobalMockAIOOperationWriteError(bool error) {
  global_mock_aio_operation_io_write_error = error;
}

void SetGlobalMockAIOOperationReadTimeout(bool timeout, uint64_t timeout_ms) {
  global_mock_aio_operation_io_read_timeout = timeout;
  global_mock_aio_operation_io_timeout_ms = timeout_ms;
}

void SetGlobalMockAIOOperationWriteTimeout(bool timeout, uint64_t timeout_ms) {
  global_mock_aio_operation_io_write_timeout = timeout;
  global_mock_aio_operation_io_timeout_ms = timeout_ms;
}

// Functions to set global error injection flags for MockIOAdaptor
void SetGlobalMockIOReadError(bool error) { global_mock_io_read_error = error; }

void SetGlobalMockIOWriteError(bool error) {
  global_mock_io_write_error = error;
}

void SetGlobalMockIOReadTimeout(bool timeout, uint64_t timeout_ms) {
  global_mock_io_read_timeout = timeout;
  global_mock_io_timeout_ms = timeout_ms;
}

void SetGlobalMockIOWriteTimeout(bool timeout, uint64_t timeout_ms) {
  global_mock_io_write_timeout = timeout;
  global_mock_io_timeout_ms = timeout_ms;
}

void SetGlobalMockIOBatchReadError(bool error) {
  global_mock_io_batch_read_error = error;
}

void SetGlobalMockIOBatchWriteError(bool error) {
  global_mock_io_batch_write_error = error;
}

void SetGlobalMockIOBatchReadTimeout(bool timeout, uint64_t timeout_ms) {
  global_mock_io_batch_read_timeout = timeout;
  global_mock_io_timeout_ms = timeout_ms;
}

void SetGlobalMockIOBatchWriteTimeout(bool timeout, uint64_t timeout_ms) {
  global_mock_io_batch_write_timeout = timeout;
  global_mock_io_timeout_ms = timeout_ms;
}

}  // namespace io

}  // namespace vllm_kv_cache

}  // namespace vineyard
