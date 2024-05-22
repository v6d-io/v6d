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
#ifndef MODULES_RDMA_RDMA_CLIENT_H_
#define MODULES_RDMA_RDMA_CLIENT_H_

#include <rdma/fabric.h>

#include "rdma.h"

namespace vineyard {

class RDMAClient : public IRDMA {
 public:

  Status Send(void *buf, size_t size, void* mr_desc, void *ctx);

  Status Recv(void *buf, size_t size, void* mr_desc, void *ctx);

  Status Read(void *buf, size_t size, uint64_t remote_address, uint64_t key, void* mr_desc, void *ctx);

  Status Write(void *buf, size_t size, uint64_t remote_address, uint64_t key, void* mr_desc, void *ctx);

  static Status Make(std::shared_ptr<RDMAClient> &ptr, std::string server_address, int port);

  static Status Make(std::shared_ptr<RDMAClient> &ptr, fi_info *hints, std::string server_address, int port);

  Status RegisterMemory(void *address, size_t size, uint64_t &rkey, void* &mr_desc);

  Status Connect();

  Status Close();

 private:
  RDMAClient() = default;

  bool IsClient() override {
    return true;
  };

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
  fid_mr *mr = NULL;
  fi_addr_t remote_fi_addr = FI_ADDR_UNSPEC;
};

}  // namespace vineyard

#endif // MODULES_RDMA_RDMA_CLIENT_H_
