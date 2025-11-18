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

#include "vllm-kv-cache/src/io/aio_operations.h"
#include <libaio.h>
#include "common/util/logging.h"

namespace vineyard {

namespace vllm_kv_cache {

namespace io {

int RealAIOOperations::io_setup(int maxevents, io_context_t* ctx_idp) {
  return ::io_setup(maxevents, ctx_idp);
}

int RealAIOOperations::io_submit(io_context_t ctx_id, int64_t nr,
                                 struct iocb* ios[]) {
  return ::io_submit(ctx_id, nr, ios);
}

int RealAIOOperations::io_getevents(io_context_t ctx_id, int64_t min_nr,
                                    int64_t nr, struct io_event* events,
                                    struct timespec* timeout) {
  return ::io_getevents(ctx_id, min_nr, nr, events, timeout);
}

int RealAIOOperations::io_destroy(io_context_t ctx_id) {
  return ::io_destroy(ctx_id);
}

void RealAIOOperations::io_prep_pread(struct iocb* iocb, int fd, void* buf,
                                      size_t count, int64_t offset) {
  ::io_prep_pread(iocb, fd, buf, count, offset);
}

void RealAIOOperations::io_prep_pwrite(struct iocb* iocb, int fd, void* buf,
                                       size_t count, int64_t offset) {
  ::io_prep_pwrite(iocb, fd, buf, count, offset);
}

}  // namespace io

}  // namespace vllm_kv_cache

}  // namespace vineyard
