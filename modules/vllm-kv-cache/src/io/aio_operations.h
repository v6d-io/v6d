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

#ifndef MODULES_VLLM_KV_CACHE_SRC_IO_AIO_OPERATIONS_H_
#define MODULES_VLLM_KV_CACHE_SRC_IO_AIO_OPERATIONS_H_

#include <libaio.h>

namespace vineyard {

namespace vllm_kv_cache {

namespace io {

class IAIOOperations {
 public:
  virtual ~IAIOOperations() = default;

  virtual int io_setup(int maxevents, io_context_t* ctx_idp) = 0;
  virtual int io_submit(io_context_t ctx_id, int64_t nr,
                        struct iocb* ios[]) = 0;
  virtual int io_getevents(io_context_t ctx_id, int64_t min_nr, int64_t nr,
                           struct io_event* events,
                           struct timespec* timeout) = 0;
  virtual int io_destroy(io_context_t ctx_id) = 0;

  virtual void io_prep_pread(struct iocb* iocb, int fd, void* buf, size_t count,
                             int64_t offset) = 0;
  virtual void io_prep_pwrite(struct iocb* iocb, int fd, void* buf,
                              size_t count, int64_t offset) = 0;
};

class RealAIOOperations : public IAIOOperations {
 public:
  virtual ~RealAIOOperations() = default;

  int io_setup(int maxevents, io_context_t* ctx_idp) override;
  int io_submit(io_context_t ctx_id, int64_t nr, struct iocb* ios[]) override;
  int io_getevents(io_context_t ctx_id, int64_t min_nr, int64_t nr,
                   struct io_event* events, struct timespec* timeout) override;
  int io_destroy(io_context_t ctx_id) override;

  void io_prep_pread(struct iocb* iocb, int fd, void* buf, size_t count,
                     int64_t offset) override;
  void io_prep_pwrite(struct iocb* iocb, int fd, void* buf, size_t count,
                      int64_t offset) override;
};

}  // namespace io

}  // namespace vllm_kv_cache

}  // namespace vineyard

#endif  // MODULES_VLLM_KV_CACHE_SRC_IO_AIO_OPERATIONS_H_
