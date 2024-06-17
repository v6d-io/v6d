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

#ifdef VINEYARD_WITH_RDMA

#include "common/rdma/rdma.h"
#include "common/rdma/util.h"

namespace vineyard {

size_t IRDMA::max_register_size_ = 0;
static int default_port = 12345;
constexpr size_t min_l_size = 1;
constexpr size_t max_r_size = 8;

size_t IRDMA::GetMaxRegisterSize() {
  if (max_register_size_ == 0) {
    fi_info* hints = fi_allocinfo();
    if (!hints) {
      return 0;
    }

    hints->caps =
        FI_MSG | FI_RMA | FI_WRITE | FI_REMOTE_WRITE | FI_READ | FI_REMOTE_READ;
    hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
    hints->mode = FI_CONTEXT;
    hints->domain_attr->threading = FI_THREAD_DOMAIN;
    hints->addr_format = FI_FORMAT_UNSPEC;
    hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_ENDPOINT |
                                  FI_MR_ALLOCATED | FI_MR_PROV_KEY |
                                  FI_MR_VIRT_ADDR | FI_MR_RAW;
    hints->tx_attr->tclass = FI_TC_BULK_DATA;
    hints->ep_attr->type = FI_EP_MSG;
    hints->fabric_attr = new fi_fabric_attr;
    memset(hints->fabric_attr, 0, sizeof *(hints->fabric_attr));
    hints->fabric_attr->prov_name = strdup("verbs");

    fi_info* fi = nullptr;
    fid_fabric* fabric = nullptr;
    fid_domain* domain = nullptr;
    if (fi_getinfo(VINEYARD_FIVERSION, nullptr,
                   std::to_string(default_port).c_str(), 0, hints, &(fi))) {
      FreeInfo(hints);
      return 0;
    }

    if (fi_fabric(fi->fabric_attr, &fabric, NULL)) {
      FreeInfo(fi);
      FreeInfo(hints);
      return 0;
    }

    if (fi_domain(fabric, fi, &domain, NULL)) {
      CloseResource(fabric, "fabric");
      FreeInfo(fi);
      FreeInfo(hints);
      return 0;
    }

    max_register_size_ = GetMaxRegisterSizeImpl(domain);

    CloseResource(domain, "domain");
    CloseResource(fabric, "fabric");
    FreeInfo(fi);
    FreeInfo(hints);
  }
  return max_register_size_;
}

size_t IRDMA::GetMaxRegisterSizeImpl(fid_domain* domain) {
  size_t l_size = min_l_size;
  size_t r_size = max_r_size;
  fid_mr* mr = nullptr;
  void* mr_desc = nullptr;
  uint64_t rkey = 0;
  void* buffer = nullptr;
  size_t size_ = (r_size + l_size) / 2;
  size_t max_size = 0;

  while (l_size < r_size - 1) {
    buffer = malloc(size_ * 1024 * 1024 * 1024);
    if (buffer == nullptr) {
      return max_size;
    }
    if (RegisterMemory(&mr, domain, buffer, size_ * 1024 * 1024 * 1024, rkey,
                       mr_desc)
            .ok()) {
      LOG(INFO) << "Register memory size: " << size_ << "GB";
      max_size = size_ * 1024 * 1024 * 1024;
      CloseResource(mr, "memory region");
      free(buffer);
      l_size = size_;
      size_ = (size_ + r_size) / 2;
    } else {
      free(buffer);
      r_size = size_;
      size_ = (size_ + l_size) / 2;
    }
  }
  return max_size;
}

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
  LOG(INFO) << "Try to register memory region: size=" << size;

  int ret = fi_mr_regattr(domain, &mr_attr, FI_HMEM_DEVICE_ONLY, mr);
  CHECK_ERROR(!ret, "Failed to register memory region:" + std::to_string(ret));

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
