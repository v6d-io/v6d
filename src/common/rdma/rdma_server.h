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
#ifndef MODULES_RDMA_RDMA_SERVER_H_
#define MODULES_RDMA_RDMA_SERVER_H_

#include <map>

#include <rdma/fabric.h>

#include "rdma.h"
#include "util.h"

namespace vineyard {

class RDMAServer : public IRDMA {
 public:
  RDMAServer() = default;

  static Status Make(std::shared_ptr<RDMAServer> &ptr, int port);

  static Status Make(std::shared_ptr<RDMAServer> &ptr, fi_info *hints, int port);

  Status Send(uint64_t clientID, void* buf, size_t size);

  Status Recv(uint64_t clientID, void* buf, size_t size);

  Status Read(uint64_t clientID, void *buf, size_t size, uint64_t remote_address);

  Status Write(uint64_t clientID, void *buf, size_t size, uint64_t remote_address);

  Status GetVineyardBufferContext(uint64_t key, VineyardBufferContext &ctx) {
    auto iter = buffer_map.find(key);
    if (iter == buffer_map.end()) {
      return Status::Invalid("Failed to find buffer context");
    }
    ctx = iter->second;
    return Status::OK();
  }

  Status WaitConnect();

  Status Close();

 private:

  bool IsClient() override {
    return false;
  };

  fid_pep *pep = NULL;
  std::map<uint64_t, VineyardBufferContext> buffer_map;
  int port;

  fi_info *fi = NULL;
  fid_fabric *fabric = NULL;
  fi_eq_attr eq_attr = { 0 };
  fid_eq *eq = NULL;
  fid_domain *domain = NULL;
  fi_cq_attr cq_attr= { 0 };
  fid_cq *rxcq = NULL, *txcq = NULL;
  fid_ep *ep = NULL;
  void* rx_msg_buffer, *tx_msg_buffer;
  uint64_t rx_msg_size = 1024, tx_msg_size = 1024;
  uint64_t rx_msg_key = 0, tx_msg_key = 0;
  void *rx_msg_mr_desc = NULL, *tx_msg_mr_desc = NULL;
  void *data_mem_desc = NULL;
  fid_mr *mr = NULL;
  fi_addr_t remote_fi_addr = FI_ADDR_UNSPEC;
};

}  // namespace vineyard

#endif  // MODULES_RDMA_RDMA_SERVER_H_
