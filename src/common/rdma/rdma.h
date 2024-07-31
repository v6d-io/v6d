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

#ifndef SRC_COMMON_RDMA_RDMA_H_
#define SRC_COMMON_RDMA_RDMA_H_

#include <map>
#include <string>
#include <vector>

#include "common/rdma/util.h"
#include "common/util/status.h"

#if defined(__linux__)
#include "libfabric/include/rdma/fabric.h"
#include "libfabric/include/rdma/fi_domain.h"
#include "libfabric/include/rdma/fi_endpoint.h"
#include "libfabric/include/rdma/fi_eq.h"
#include "libfabric/include/rdma/fi_rma.h"
#endif  // defined(__linux__)

namespace vineyard {

class IRDMA {
 public:
#if defined(__linux__)
  static Status Send(fid_ep* ep, fi_addr_t remote_fi_addr, void* buf,
                     size_t size, void* mr_desc, void* ctx);

  static Status Recv(fid_ep* ep, fi_addr_t remote_fi_addr, void* buf,
                     size_t size, void* mr_desc, void* ctx);

  static Status Read(fid_ep* ep, fi_addr_t remote_fi_addr, void* buf,
                     size_t size, uint64_t remote_address, uint64_t key,
                     void* mr_desc, void* ctx);

  static Status Write(fid_ep* ep, fi_addr_t remote_fi_addr, void* buf,
                      size_t size, uint64_t remote_address, uint64_t key,
                      void* mr_desc, void* ctx);

  static Status RegisterMemory(fid_mr** mr, fid_domain* domain, void* address,
                               size_t size, uint64_t& rkey, void*& mr_desc);

  static int GetCompletion(fid_cq* cq, int timeout, void** context);

  static void FreeInfo(fi_info* info);

  template <typename FIDType>
  static Status CloseResource(FIDType* res, const char* resource_name) {
    if (res) {
      int ret = fi_close(&(res)->fid);
      if (ret != FI_SUCCESS) {
        return Status::Wrap(Status::IOError(),
                            "Failed to close resource (" +
                                std::string(resource_name) +
                                "): " + std::string(fi_strerror(-ret)));
      }
    }
    return Status::OK();
  }

  template <typename FIDType>
  static Status CloseResourcesInVector(std::vector<FIDType*>& vec,
                                       const char* resource_name) {
    for (auto& res : vec) {
      RETURN_ON_ERROR(CloseResource(res, resource_name));
    }
    vec.clear();
    return Status::OK();
  }

  template <typename K, typename FIDType>
  static Status CloseResourcesInMap(std::map<K, FIDType*>& mapping,
                                    const char* resource_name) {
    for (auto iter = mapping.begin(); iter != mapping.end(); ++iter) {
      RETURN_ON_ERROR(CloseResource(iter->second, resource_name));
    }
    mapping.clear();
    return Status::OK();
  }

  static size_t GetMaxRegisterSizeImpl(void* addr, size_t min_size,
                                       size_t max_size, fid_domain* domain);

  static size_t max_register_size_;
#endif  // defined(__linux__)
};

}  // namespace vineyard

#endif  // SRC_COMMON_RDMA_RDMA_H_
