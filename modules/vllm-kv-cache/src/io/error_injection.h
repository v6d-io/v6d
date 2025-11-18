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

#ifndef MODULES_VLLM_KV_CACHE_SRC_IO_ERROR_INJECTION_H_
#define MODULES_VLLM_KV_CACHE_SRC_IO_ERROR_INJECTION_H_

#include <stdint.h>

namespace vineyard {

namespace vllm_kv_cache {

namespace io {

// Global error injection flags for MockAIOOperations
extern bool global_mock_aio_operation_io_setup_error;
extern bool global_mock_aio_operation_io_submit_timeout;
extern bool global_mock_aio_operation_io_submit_error;
extern bool global_mock_aio_operation_io_submit_part_processed;
extern int64_t global_mock_aio_operation_io_submit_max_processed;
extern bool global_mock_aio_operation_io_getevents_timeout;
extern bool global_mock_aio_operation_io_getevents_error;
extern bool global_mock_aio_operation_io_getevents_no_events;
extern uint64_t global_mock_aio_operation_io_submit_timeout_ms;
extern uint64_t global_mock_aio_operation_io_getevents_timeout_ms;
extern int global_mock_aio_operation_io_setup_error_code;
extern int global_mock_aio_operation_io_submit_error_code;
extern int global_mock_aio_operation_io_getevents_error_code;

extern bool global_mock_aio_operation_io_read_error;
extern bool global_mock_aio_operation_io_write_error;
extern bool global_mock_aio_operation_io_read_timeout;
extern bool global_mock_aio_operation_io_write_timeout;
extern uint64_t global_mock_aio_operation_io_timeout_ms;

// Global error injection flags for MockIOAdaptor
extern bool global_mock_io_read_error;
extern bool global_mock_io_write_error;
extern bool global_mock_io_read_timeout;
extern bool global_mock_io_write_timeout;
extern bool global_mock_io_batch_read_error;
extern bool global_mock_io_batch_write_error;
extern bool global_mock_io_batch_read_timeout;
extern bool global_mock_io_batch_write_timeout;
extern uint64_t global_mock_io_timeout_ms;

// Functions to set global error injection flags for MockAIOOperations
void SetGlobalMockAIOOperationSetupError(bool error);
void SetGlobalMockAIOOperationSubmitTimeout(bool timeout,
                                            uint64_t timeout_ms = 1000);
void SetGlobalMockAIOOperationSubmitError(bool error);
void SetGlobalMockAIOOperationSubmitMaxProcessedPerCall(
    bool is_part_processed, uint64_t max_processed = 3);
void SetGlobalMockAIOOperationGetEventsTimeout(bool timeout,
                                               uint64_t timeout_ms = 1000);
void SetGlobalMockAIOOperationGetEventsError(bool error);
void SetGlobalMockAIOOperationGetEventsNoEvents(bool no_events);
void SetGlobalMockAIOOperationReadError(bool error);
void SetGlobalMockAIOOperationWriteError(bool error);
void SetGlobalMockAIOOperationReadTimeout(bool timeout,
                                          uint64_t timeout_ms = 1000);
void SetGlobalMockAIOOperationWriteTimeout(bool timeout,
                                           uint64_t timeout_ms = 1000);

// Functions to set global error injection flags for MockIOAdaptor
void SetGlobalMockIOReadError(bool error);
void SetGlobalMockIOWriteError(bool error);
void SetGlobalMockIOReadTimeout(bool timeout, uint64_t timeout_ms = 1000);
void SetGlobalMockIOWriteTimeout(bool timeout, uint64_t timeout_ms = 1000);
void SetGlobalMockIOBatchReadError(bool error);
void SetGlobalMockIOBatchWriteError(bool error);
void SetGlobalMockIOBatchReadTimeout(bool timeout, uint64_t timeout_ms = 1000);
void SetGlobalMockIOBatchWriteTimeout(bool timeout, uint64_t timeout_ms = 1000);

}  // namespace io

}  // namespace vllm_kv_cache

}  // namespace vineyard

#endif  // MODULES_VLLM_KV_CACHE_SRC_IO_ERROR_INJECTION_H_
