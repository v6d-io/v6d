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
#include <mutex>

#include <rdma/fabric.h>

#include "rdma.h"
#include "util.h"

namespace vineyard {

class RDMAServer : public IRDMA {
 public:
  RDMAServer() = default;

  static Status Make(std::shared_ptr<RDMAServer> &ptr, int port);

  static Status Make(std::shared_ptr<RDMAServer> &ptr, fi_info *hints, int port);

  Status Send(uint64_t clientID, void* buf, size_t size, void* ctx);

  Status Recv(uint64_t clientID, void* buf, size_t size, void* ctx);

  Status Send(void *ep, void* buf, size_t size, void* ctx);

  Status Recv(void *ep, void* buf, size_t size, void* ctx);

  Status Read(uint64_t clientID, void *buf, size_t size, uint64_t remote_address, uint64_t rkey, void *mr_desc, void* ctx);

  Status Write(uint64_t clientID, void *buf, size_t size, uint64_t remote_address, uint64_t rkey, void *mr_desc, void* ctx);

  Status GetTXFreeMsgBuffer(void *&buffer);

  Status GetRXFreeMsgBuffer(void *&buffer);

  Status GetRXCompletion(int timeout, void **context);

  Status GetTXCompletion(int timeout, void **context);

  Status RegisterMemory(RegisterMemInfo &memInfo);

  // TODO: delete in the future.
  Status RegisterMemory(fid_mr **mr, void *address, size_t size, uint64_t &rkey, void* &mr_desc);

  Status GetEp(uint64_t ep_token, fid_ep *&ep) {
    auto iter = ep_map_.find(ep_token);
    if (iter == ep_map_.end()) {
      return Status::Invalid("Failed to find buffer context");
    }
    ep = iter->second;
    return Status::OK();
  }

  Status WaitConnect(void *&rdma_conn_handle);

  Status Close();

  Status AddClient(uint64_t clientID, void *ep);

  Status RemoveClient(uint64_t ep_token);

 private:

  bool IsClient() override {
    return false;
  };

  fid_pep *pep = NULL;
  std::mutex ep_map_mutex_;
  std::map<uint64_t, fid_ep*> ep_map_;
  int port;

  fi_info *fi = NULL;
  fid_fabric *fabric = NULL;
  fi_eq_attr eq_attr = { 0 };
  fid_eq *eq = NULL;
  fid_domain *domain = NULL;
  fi_cq_attr cq_attr= { 0 };
  fid_cq *rxcq = NULL, *txcq = NULL;
  uint64_t mem_key;
  void* rx_msg_buffer, *tx_msg_buffer;
  uint64_t rx_msg_size = 1024, tx_msg_size = 1024;
  uint64_t rx_msg_key = 0, tx_msg_key = 0;
  void *rx_msg_mr_desc = NULL, *tx_msg_mr_desc = NULL;
  void *data_mem_desc = NULL;
  fid_mr *tx_mr = NULL, *rx_mr = NULL;
  std::vector<fid_mr *> mr_array;
  fi_addr_t remote_fi_addr = FI_ADDR_UNSPEC;
  RegisterMemInfo info;
};

}  // namespace vineyard

#endif  // MODULES_RDMA_RDMA_SERVER_H_
