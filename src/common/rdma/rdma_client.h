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
#ifndef SRC_COMMON_RDMA_RDMA_CLIENT_H_
#define SRC_COMMON_RDMA_RDMA_CLIENT_H_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/rdma/rdma.h"
#include "common/rdma/util.h"
#if defined(__linux__)
#include "libfabric/include/rdma/fabric.h"
#include "libfabric/include/rdma/fi_domain.h"
#endif

namespace vineyard {

class RDMAClient : public IRDMA {
 public:
  using rdma_opt_t =
      std::function<Status(void* buf, size_t size, uint64_t remote_address,
                           uint64_t key, void* mr_desc, void* ctx)>;

  RDMAClient() = default;

  Status Send(void* buf, size_t size, void* ctx);

  Status Recv(void* buf, size_t size, void* ctx);

  Status Read(void* buf, size_t size, uint64_t remote_address, uint64_t key,
              void* mr_desc, void* ctx);

  Status Write(void* buf, size_t size, uint64_t remote_address, uint64_t key,
               void* mr_desc, void* ctx);

  Status RegisterMemory(RegisterMemInfo& memInfo);

  Status DeregisterMemory(RegisterMemInfo& memInfo);

  Status Connect();

  Status Close();

  Status Stop();

  Status SendMemInfoToServer(void* buffer, uint64_t size);

  Status GetTXFreeMsgBuffer(void*& buffer);

  Status GetRXFreeMsgBuffer(void*& buffer);

  Status GetRXCompletion(int timeout, void** context);

  Status GetTXCompletion(int timeout, void** context);

  size_t GetMaxTransferBytes();

  size_t GetClientMaxRegisterSize(void* addr = nullptr, size_t min_size = 8192,
                                  size_t max_size = 64UL * 1024 * 1024 * 1024);

 private:
#if defined(__linux__)
  static Status Make(std::shared_ptr<RDMAClient>& ptr,
                     RDMARemoteNodeInfo& info);

  Status RegisterMemory(fid_mr** mr, void* address, size_t size, uint64_t& rkey,
                        void*& mr_desc);

  fi_info* fi = NULL;
  fid_fabric* fabric = NULL;
  fid_eq* eq = NULL;
  fid_domain* domain = NULL;
  fid_cq *rxcq = NULL, *txcq = NULL;
  fid_ep* ep = NULL;
  char *rx_msg_buffer, *tx_msg_buffer;

  // client just need one tx and rx buffer
  uint64_t rx_msg_size = sizeof(VineyardMsg), tx_msg_size = sizeof(VineyardMsg);
  uint64_t rx_msg_key = 0, tx_msg_key = 0;
  void *rx_msg_mr_desc = NULL, *tx_msg_mr_desc = NULL;
  fid_mr *tx_mr = NULL, *rx_mr = NULL;
  std::vector<fid_mr*> mr_array;
  fi_addr_t remote_fi_addr = FI_ADDR_UNSPEC;

  RDMA_STATE state = INIT;
#endif  // defined(__linux__)

  friend class RDMAClientCreator;
};

class RDMAClientCreator {
 public:
  static Status Create(std::shared_ptr<RDMAClient>& ptr,
                       std::string server_address, int port);

  static Status Release(std::string rdma_endpoint);

  static Status Clear();

 private:
#if defined(__linux__)
  RDMAClientCreator() = delete;

  static Status Create(std::shared_ptr<RDMAClient>& ptr, fi_info* hints,
                       std::string server_address, int port);

  static Status CreateRDMARemoteNodeInfo(RDMARemoteNodeInfo& info,
                                         fi_info* hints,
                                         std::string server_address, int port);

  static Status CreateRDMARemoteNodeInfo(RDMARemoteNodeInfo& info,
                                         std::string server_address, int port);

  static std::map<std::string, RDMARemoteNodeInfo> servers_;
  static std::mutex servers_mtx_;
#endif  // defined(__linux__)
};

}  // namespace vineyard

#endif  // SRC_COMMON_RDMA_RDMA_CLIENT_H_
