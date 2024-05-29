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

#ifndef MODULES_RDMA_RDMA_H_
#define MODULES_RDMA_RDMA_H_

#include <map>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>

#include "common/util/status.h"

namespace vineyard {

class IRDMA {
 public:
  Status Send(fid_ep *ep, fi_addr_t remote_fi_addr, fid_cq *txcq, void *buf, size_t size, void* mr_desc, void *ctx);

  Status Recv(fid_ep *ep, fi_addr_t remote_fi_addr, fid_cq *rxcq, void *buf, size_t size, void* mr_desc, void *ctx);

  Status Read(fid_ep *ep, fi_addr_t remote_fi_addr, fid_cq *rxcq, void *buf, size_t size, uint64_t remote_address, uint64_t key, void* mr_desc, void *ctx);

  Status Write(fid_ep *ep, fi_addr_t remote_fi_addr, fid_cq *txcq, void *buf, size_t size, uint64_t remote_address, uint64_t key, void* mr_desc, void *ctx);

  Status RegisterMemory(fi_info *fi, fid_mr **mr, fid_domain *domain, void *address, size_t size, uint64_t &rkey, void* &mr_desc);

  virtual Status Close() = 0;

  Status GetCompletion(fi_addr_t remote_fi_addr, fid_cq *cq, uint64_t *cur, uint64_t total, int timeout, void **context);

  Status Release();

  int Progress(fid_cq *cq, uint64_t total, uint64_t *cq_cntr);

  virtual bool IsClient() = 0;
};

}  // namespace vineyard

#endif  // MODULES_RDMA_RDMA_H_
