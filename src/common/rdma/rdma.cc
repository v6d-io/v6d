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

#ifndef VINEYARD_WITHOUT_RDMA

#include "common/rdma/rdma.h"
#include "common/rdma/util.h"

namespace vineyard {

Status IRDMA::RegisterMemory(fid_mr** mr, fid_domain* domain, void* address,
                             size_t size, uint64_t& rkey, void*& mr_desc) {
  struct fi_mr_attr mr_attr = {0};
  struct iovec iov = {0};
  iov.iov_base = address;
  iov.iov_len = size;
  mr_attr.mr_iov = &iov;
  mr_attr.iov_count = 1;
  mr_attr.access = FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE;
  mr_attr.offset = 0;
  mr_attr.iface = FI_HMEM_SYSTEM;
  mr_attr.context = NULL;

  int ret = fi_mr_regattr(domain, &mr_attr, FI_HMEM_DEVICE_ONLY, mr);
  CHECK_ERROR(!ret, "Failed to register memory region");

  mr_desc = fi_mr_desc(*mr);

  rkey = fi_mr_key(*mr);

  return Status::OK();
}

Status IRDMA::Send(fid_ep* ep, fi_addr_t remote_fi_addr, fid_cq* txcq,
                   void* buf, size_t size, void* mr_desc, void* ctx) {
  POST(fi_send, "send", ep, buf, size, mr_desc, remote_fi_addr, ctx);
}

Status IRDMA::Recv(fid_ep* ep, fi_addr_t remote_fi_addr, fid_cq* rxcq,
                   void* buf, size_t size, void* mr_desc, void* ctx) {
  POST(fi_recv, "receive", ep, buf, size, mr_desc, remote_fi_addr, ctx);
}

Status IRDMA::Read(fid_ep* ep, fi_addr_t remote_fi_addr, fid_cq* rxcq,
                   void* buf, size_t size, uint64_t remote_address,
                   uint64_t key, void* mr_desc, void* ctx) {
  POST(fi_read, "read", ep, buf, size, mr_desc, remote_fi_addr, remote_address,
       key, ctx);
}

Status IRDMA::Write(fid_ep* ep, fi_addr_t remote_fi_addr, fid_cq* txcq,
                    void* buf, size_t size, uint64_t remote_address,
                    uint64_t key, void* mr_desc, void* ctx) {
  POST(fi_write, "write", ep, buf, size, mr_desc, remote_fi_addr,
       remote_address, key, ctx);
}

int IRDMA::GetCompletion(fid_cq* cq, int timeout, void** context) {
  fi_cq_err_entry err;
  timespec start, end;
  int ret;

  if (timeout > 0) {
    clock_gettime(CLOCK_REALTIME, &start);
  }

  while (true) {
    ret = fi_cq_read(cq, &err, 1);
    if (ret > 0) {
      break;
    } else if (ret < 0 && ret != -FI_EAGAIN) {
      return ret;
    } else if (timeout > 0) {
      clock_gettime(CLOCK_REALTIME, &end);
      if ((end.tv_sec - start.tv_sec) * 1000 +
              (end.tv_nsec - start.tv_nsec) / 1000000 >
          timeout) {
        return -FI_ETIMEDOUT;
      }
    }
  }
  if (context) {
    *context = err.op_context;
  }

  return 0;
}

void IRDMA::FreeInfo(fi_info* info) {
  if (info) {
    fi_freeinfo(info);
  }
}

}  // namespace vineyard

#endif
