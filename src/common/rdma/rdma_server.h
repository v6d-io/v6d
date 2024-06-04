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
#ifndef SRC_COMMON_RDMA_RDMA_SERVER_H_
#define SRC_COMMON_RDMA_RDMA_SERVER_H_

#ifdef VINEYARD_WITH_RDMA

#include <rdma/fabric.h>

#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "common/rdma/rdma.h"
#include "common/rdma/util.h"

namespace vineyard {

class RDMAServer : public IRDMA {
 public:
  RDMAServer() = default;

  ~RDMAServer() {}

  static Status Make(std::shared_ptr<RDMAServer>& ptr, int port);

  static Status Make(std::shared_ptr<RDMAServer>& ptr, fi_info* hints,
                     int port);

  Status Send(uint64_t clientID, void* buf, size_t size, void* ctx);

  Status Recv(uint64_t clientID, void* buf, size_t size, void* ctx);

  Status Send(void* ep, void* buf, size_t size, void* ctx);

  Status Recv(void* ep, void* buf, size_t size, void* ctx);

  Status Read(uint64_t clientID, void* buf, size_t size,
              uint64_t remote_address, uint64_t rkey, void* mr_desc, void* ctx);

  Status Write(uint64_t clientID, void* buf, size_t size,
               uint64_t remote_address, uint64_t rkey, void* mr_desc,
               void* ctx);

  Status GetTXFreeMsgBuffer(void*& buffer);

  Status GetRXFreeMsgBuffer(void*& buffer);

  Status GetRXCompletion(int timeout, void** context);

  Status GetTXCompletion(int timeout, void** context);

  Status RegisterMemory(RegisterMemInfo& memInfo);

  Status RegisterMemory(fid_mr** mr, void* address, size_t size, uint64_t& rkey,
                        void*& mr_desc);

  Status GetEp(uint64_t ep_token, fid_ep*& ep) {
    auto iter = ep_map_.find(ep_token);
    if (iter == ep_map_.end()) {
      return Status::Invalid("Failed to find buffer context");
    }
    ep = iter->second;
    return Status::OK();
  }

  Status WaitConnect(uint64_t& rdma_conn_id);

  Status Close() override;

  Status Stop() {
    state = STOPED;
    return Status::OK();
  }

  Status AddClient(uint64_t& rdma_conn_id, void* ep);

  Status CloseConnection(uint64_t rdma_conn_id);

  bool IsStopped();

  Status ReleaseRXBuffer(void* buffer);

  Status ReleaseTXBuffer(void* buffer);

 private:
  Status RemoveClient(uint64_t ep_token);

  Status RemoveClient(fid_ep* ep);

  int FindEmptySlot(uint64_t* bitmaps, int bitmap_num) {
    for (int i = 0; i < bitmap_num; i++) {
      if (bitmaps[i] != 0) {
        int index = ffsll(bitmaps[i]) - 1;
        bitmaps[i] &= ~(1 << index);
        return i * 64 + index;
      }
    }
    return -1;
  }

  fid_pep* pep = NULL;
  std::mutex ep_map_mutex_;
  std::map<uint64_t, fid_ep*> ep_map_;
  int port;

  fi_info* fi = NULL;
  fid_fabric* fabric = NULL;
  fi_eq_attr eq_attr = {0};
  fid_eq* eq = NULL;
  fid_domain* domain = NULL;
  fi_cq_attr cq_attr = {0};
  fid_cq *rxcq = NULL, *txcq = NULL;

  std::mutex rx_msg_buffer_mutex_, tx_msg_buffer_mutex_;
  char *rx_msg_buffer, *tx_msg_buffer;
  uint64_t *rx_buffer_bitmaps, *tx_buffer_bitmaps;
  int rx_bitmap_num = 0, tx_bitmap_num = 0;
  uint64_t rx_msg_size = 8192, tx_msg_size = 8192;

  uint64_t rx_msg_key = 0, tx_msg_key = 0;
  void *rx_msg_mr_desc = NULL, *tx_msg_mr_desc = NULL;
  void* data_mem_desc = NULL;
  fid_mr *tx_mr = NULL, *rx_mr = NULL;
  std::vector<fid_mr*> mr_array;
  fi_addr_t remote_fi_addr = FI_ADDR_UNSPEC;

  RDMA_STATE state = INIT;
  uint64_t current_conn_id = 0;
};

}  // namespace vineyard

#endif

#endif  // SRC_COMMON_RDMA_RDMA_SERVER_H_
