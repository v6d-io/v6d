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

#include <memory>
#include <string>

#include "common/rdma/rdma_server.h"
#include "common/rdma/util.h"
#include "common/util/status.h"
#if defined(__linux__)
#include "libfabric/include/rdma/fabric.h"
#include "libfabric/include/rdma/fi_cm.h"
#include "libfabric/include/rdma/fi_domain.h"
#include "libfabric/include/rdma/fi_endpoint.h"
#include "libfabric/include/rdma/fi_eq.h"
#include "libfabric/include/rdma/fi_rma.h"
#endif  // defined(__linux__)

namespace vineyard {

#if defined(__linux__)
Status RDMAServer::Make(std::shared_ptr<RDMAServer>& ptr, int port,
                        std::string host) {
  fi_info* hints = fi_allocinfo();
  if (!hints) {
    return Status::Invalid("Failed to allocate fabric info.");
  }

  hints->caps =
      FI_MSG | FI_RMA | FI_WRITE | FI_REMOTE_WRITE | FI_READ | FI_REMOTE_READ;
  hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
  hints->mode = FI_CONTEXT;
  hints->domain_attr->threading = FI_THREAD_DOMAIN;
  hints->addr_format = FI_FORMAT_UNSPEC;
  hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_ENDPOINT | FI_MR_ALLOCATED |
                                FI_MR_PROV_KEY | FI_MR_VIRT_ADDR | FI_MR_RAW;
  hints->tx_attr->tclass = FI_TC_BULK_DATA;
  hints->ep_attr->type = FI_EP_MSG;
  hints->fabric_attr = new fi_fabric_attr;
  memset(hints->fabric_attr, 0, sizeof *(hints->fabric_attr));
  hints->src_addr = nullptr;
  hints->src_addrlen = 0;
  hints->dest_addr = nullptr;
  hints->dest_addrlen = 0;
  hints->fabric_attr->prov_name = strdup("verbs");

  Status status = Make(ptr, hints, port, host);
  IRDMA::FreeInfo(hints, true);
  return status;
}

Status RDMAServer::Make(std::shared_ptr<RDMAServer>& ptr, fi_info* hints,
                        int port, std::string host) {
  if (!hints) {
    return Status::Invalid("Invalid fabric hints info.");
  }

  ptr = std::make_shared<RDMAServer>();

  if (!host.empty()) {
    uint64_t flags = FI_SOURCE;
    CHECK_ERROR(
        fi_getinfo(VINEYARD_FIVERSION, host.c_str(),
                   std::to_string(port).c_str(), flags, hints, &(ptr->fi)),
        "fi_getinfo failed.");
  } else {
    CHECK_ERROR(fi_getinfo(VINEYARD_FIVERSION, NULL,
                           std::to_string(port).c_str(), 0, hints, &(ptr->fi)),
                "fi_getinfo failed.");
  }
  if (ptr->fi != nullptr && ptr->fi->nic != nullptr) {
    std::cout << "open device name:" << ptr->fi->nic->device_attr->name
              << std::endl;
  }

  CHECK_ERROR(fi_fabric(ptr->fi->fabric_attr, &ptr->fabric, NULL),
              "fi_fabric failed.");

  ptr->eq_attr.wait_obj = FI_WAIT_UNSPEC;
  CHECK_ERROR(fi_eq_open(ptr->fabric, &ptr->eq_attr, &ptr->eq, NULL),
              "fi_eq_open failed.");

  CHECK_ERROR(fi_passive_ep(ptr->fabric, ptr->fi, &ptr->pep, NULL),
              "fi_passive_ep failed.");

  CHECK_ERROR(fi_pep_bind(ptr->pep, &ptr->eq->fid, 0), "fi_pep_bind failed.");

  CHECK_ERROR(fi_domain(ptr->fabric, ptr->fi, &ptr->domain, NULL),
              "fi_domain failed.");

  memset(&ptr->cq_attr, 0, sizeof cq_attr);
  ptr->cq_attr.format = FI_CQ_FORMAT_CONTEXT;
  ptr->cq_attr.wait_obj = FI_WAIT_NONE;
  ptr->cq_attr.wait_cond = FI_CQ_COND_NONE;
  ptr->cq_attr.size = ptr->fi->rx_attr->size;
  CHECK_ERROR(fi_cq_open(ptr->domain, &ptr->cq_attr, &ptr->rxcq, NULL),
              "fi_cq_open failed.");

  ptr->cq_attr.size = ptr->fi->tx_attr->size;
  CHECK_ERROR(fi_cq_open(ptr->domain, &ptr->cq_attr, &ptr->txcq, NULL),
              "fi_cq_open failed.");

  ptr->rx_msg_buffer = new char[ptr->rx_msg_size];
  if (!ptr->rx_msg_buffer) {
    return Status::Invalid("Failed to allocate rx buffer.");
  }
  ptr->tx_msg_buffer = new char[ptr->rx_msg_size];
  if (!ptr->tx_msg_buffer) {
    return Status::Invalid("Failed to allocate tx buffer.");
  }

  int bits = ptr->rx_msg_size / sizeof(VineyardMsg);
  ptr->rx_bitmap_num = ptr->tx_bitmap_num = (bits + 63) / 64;
  ptr->rx_buffer_bitmaps = new uint64_t[ptr->rx_bitmap_num];
  if (!ptr->rx_buffer_bitmaps) {
    return Status::Invalid("Failed to allocate rx buffer bitmap.");
  }
  memset(ptr->rx_buffer_bitmaps, UINT8_MAX,
         ptr->rx_bitmap_num * sizeof(uint64_t));
  ptr->tx_buffer_bitmaps = new uint64_t[ptr->tx_bitmap_num];
  if (!ptr->tx_buffer_bitmaps) {
    return Status::Invalid("Failed to allocate tx buffer bitmap.");
  }
  memset(ptr->tx_buffer_bitmaps, UINT8_MAX,
         ptr->tx_bitmap_num * sizeof(uint64_t));

  ptr->RegisterMemory(&(ptr->rx_mr), ptr->rx_msg_buffer, ptr->rx_msg_size,
                      ptr->rx_msg_key, ptr->rx_msg_mr_desc);
  ptr->RegisterMemory(&(ptr->tx_mr), ptr->tx_msg_buffer, ptr->tx_msg_size,
                      ptr->tx_msg_key, ptr->tx_msg_mr_desc);

  ptr->port = port;

  CHECK_ERROR(fi_listen(ptr->pep), "fi_listen failed.");

  ptr->state = READY;
  return Status::OK();
}

Status RDMAServer::Close() {
  // close all registered memory regions
  RETURN_ON_ERROR(CloseResource(tx_mr, "transmit memory rigion"));
  RETURN_ON_ERROR(CloseResource(rx_mr, "receive memory rigion"));
  RETURN_ON_ERROR(
      CloseResourcesInVector(mr_array, "memory regions registered by server"));

  {
    std::lock_guard<std::mutex> lock(ep_map_mutex_);
    RETURN_ON_ERROR(CloseResourcesInMap(ep_map_, "endpoint created by server"));
  }

  {
    std::lock_guard<std::mutex> lock(wait_conn_ep_map_mutex_);
    RETURN_ON_ERROR(
        CloseResourcesInMap(wait_conn_ep_map_, "endpoint created by server"));
  }

  RETURN_ON_ERROR(CloseResource(txcq, "transmit comeple queue"));
  RETURN_ON_ERROR(CloseResource(rxcq, "receive comeple queue"));
  RETURN_ON_ERROR(CloseResource(pep, "passive endpoint"));
  RETURN_ON_ERROR(CloseResource(eq, "event queue"));

  // before closing domain and fabric, we need to close all the resources
  // bound to them, otherwise, the close operation will failed with -FI_EBUSY
  RETURN_ON_ERROR(CloseResource(domain, "domain"));
  RETURN_ON_ERROR(CloseResource(fabric, "fabric"));

  delete rx_msg_buffer;
  delete tx_msg_buffer;

  delete[] rx_buffer_bitmaps;
  delete[] tx_buffer_bitmaps;

  FreeInfo(fi, false);

  return Status::OK();
}

Status RDMAServer::PrepareConnection(VineyardEventEntry vineyard_entry) {
  // prepare new ep
  fid_ep* ep = NULL;
  fi_info* client_fi = reinterpret_cast<fi_info*>(vineyard_entry.fi);
  CHECK_ERROR(fi_endpoint(domain, client_fi, &ep, NULL), "fi_endpoint failed.");

  CHECK_ERROR(fi_ep_bind(ep, &eq->fid, 0), "fi_ep_bind eq failed.");

  CHECK_ERROR(fi_ep_bind(ep, &rxcq->fid, FI_RECV), "fi_ep_bind rxcq failed.");

  CHECK_ERROR(fi_ep_bind(ep, &txcq->fid, FI_SEND), "fi_ep_bind txcq failed.");

  CHECK_ERROR(fi_enable(ep), "fi_enable failed.");

  CHECK_ERROR(fi_accept(ep, NULL, 0), "fi_accept failed.");

  std::lock_guard<std::mutex> lock(wait_conn_ep_map_mutex_);
  wait_conn_ep_map_[&ep->fid] = ep;
  return Status::OK();
}

Status RDMAServer::FinishConnection(uint64_t& rdma_conn_id,
                                    VineyardEventEntry event) {
  fid_ep* ep = nullptr;
  {
    std::lock_guard<std::mutex> lock(wait_conn_ep_map_mutex_);
    if (wait_conn_ep_map_.find(reinterpret_cast<fid_t>(event.fid)) ==
        wait_conn_ep_map_.end()) {
      return Status::Invalid("Failed to find buffer context.");
    }
    ep = wait_conn_ep_map_[reinterpret_cast<fid_t>(event.fid)];
    wait_conn_ep_map_.erase(reinterpret_cast<fid_t>(event.fid));
  }
  AddClient(rdma_conn_id, ep);
  return Status::OK();
}

Status RDMAServer::GetEvent(VineyardEventEntry& vineyard_entry) {
  struct fi_eq_cm_entry entry;
  uint32_t event;
  while (true) {
    int rd = fi_eq_sread(eq, &event, &entry, sizeof entry, 500, 0);

    if (rd < 0 && (rd != -FI_ETIMEDOUT && rd != -FI_EAGAIN)) {
      return Status::IOError("fi_eq_sread broken. ret:" + std::to_string(rd));
    }
    if (rd == -FI_ETIMEDOUT || rd == -FI_EAGAIN) {
      if (state == STOPED) {
        return Status::Invalid("Server is stoped.");
      }
      continue;
    }
    if (event == FI_SHUTDOWN) {
      fid_ep* closed_ep = container_of(entry.fid, fid_ep, fid);
      RemoveClient(closed_ep);
      continue;
    }
    vineyard_entry.fi = entry.info;
    vineyard_entry.event_id = event;
    vineyard_entry.fid = entry.fid;
    return Status::OK();
  }
}

Status RDMAServer::AddClient(uint64_t& ep_token, void* ep) {
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  ep_token = current_conn_id++;
  ep_map_[ep_token] = reinterpret_cast<fid_ep*>(ep);
  return Status::OK();
}

Status RDMAServer::RemoveClient(uint64_t ep_token) {
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  if (ep_map_.find(ep_token) != ep_map_.end()) {
    CloseResource(ep_map_[ep_token], "client endpoint");
    ep_map_.erase(ep_token);
  }
  return Status::OK();
}

Status RDMAServer::RemoveClient(fid_ep* ep) {
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  for (auto iter = ep_map_.begin(); iter != ep_map_.end(); iter++) {
    if (iter->second == ep) {
      ep_map_.erase(iter);
      CloseResource(ep, "client endpoint");
      return Status::OK();
    }
  }
  return Status::OK();
}

Status RDMAServer::RegisterMemory(RegisterMemInfo& memInfo) {
  fid_mr* new_mr = NULL;
  RETURN_ON_ERROR(IRDMA::RegisterMemory(
      &new_mr, domain, reinterpret_cast<void*>(memInfo.address), memInfo.size,
      memInfo.rkey, memInfo.mr_desc));
  std::lock_guard<std::mutex> lock(mr_array_mutex_);
  mr_array.push_back(new_mr);
  memInfo.mr = new_mr;
  return Status::OK();
}

Status RDMAServer::RegisterMemory(fid_mr** mr, void* address, size_t size,
                                  uint64_t& rkey, void*& mr_desc) {
  return IRDMA::RegisterMemory(mr, domain, address, size, rkey, mr_desc);
}

Status RDMAServer::DeregisterMemory(RegisterMemInfo& memInfo) {
  {
    std::lock_guard<std::mutex> lock(mr_array_mutex_);
    mr_array.erase(std::remove(mr_array.begin(), mr_array.end(), memInfo.mr),
                   mr_array.end());
  }
  VINEYARD_CHECK_OK(IRDMA::CloseResource(reinterpret_cast<fid_mr*>(memInfo.mr),
                                         "memory region"));
  return Status::OK();
}

Status RDMAServer::Send(uint64_t ep_token, void* buf, size_t size, void* ctx) {
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  if (ep_map_.find(ep_token) == ep_map_.end()) {
    return Status::Invalid("Failed to find buffer context.");
  }
  fid_ep* ep = ep_map_[ep_token];
  return IRDMA::Send(ep, remote_fi_addr, buf, size, tx_msg_mr_desc, ctx);
}

Status RDMAServer::Recv(uint64_t ep_token, void* buf, size_t size, void* ctx) {
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  if (ep_map_.find(ep_token) == ep_map_.end()) {
    return Status::Invalid("Failed to find buffer context.");
  }
  fid_ep* ep = ep_map_[ep_token];
  return IRDMA::Recv(ep, remote_fi_addr, buf, size, rx_msg_mr_desc, ctx);
}

Status RDMAServer::Send(void* ep, void* buf, size_t size, void* ctx) {
  return IRDMA::Send(reinterpret_cast<fid_ep*>(ep), remote_fi_addr, buf, size,
                     tx_msg_mr_desc, ctx);
}

Status RDMAServer::Recv(void* ep, void* buf, size_t size, void* ctx) {
  return IRDMA::Recv(reinterpret_cast<fid_ep*>(ep), remote_fi_addr, buf, size,
                     rx_msg_mr_desc, ctx);
}

Status RDMAServer::Read(uint64_t ep_token, void* buf, size_t size,
                        uint64_t remote_address, uint64_t rkey, void* mr_desc,
                        void* ctx) {
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  if (ep_map_.find(ep_token) == ep_map_.end()) {
    return Status::Invalid("Failed to find buffer context.");
  }
  fid_ep* ep = ep_map_[ep_token];
  return IRDMA::Read(ep, remote_fi_addr, buf, size, remote_address, rkey,
                     mr_desc, ctx);
}

Status RDMAServer::Write(uint64_t ep_token, void* buf, size_t size,
                         uint64_t remote_address, uint64_t rkey, void* mr_desc,
                         void* ctx) {
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  if (ep_map_.find(ep_token) == ep_map_.end()) {
    return Status::Invalid("Failed to find buffer context.");
  }
  fid_ep* ep = ep_map_[ep_token];
  return IRDMA::Write(ep, remote_fi_addr, buf, size, remote_address, rkey,
                      mr_desc, ctx);
}

Status RDMAServer::GetTXFreeMsgBuffer(void*& buffer) {
  if (state == STOPED) {
    return Status::Invalid("Server is stoped.");
  }
  while (true) {
    std::lock_guard<std::mutex> lock(tx_msg_buffer_mutex_);
    int index = FindEmptySlot(tx_buffer_bitmaps, tx_bitmap_num);
    if (index == -1) {
      usleep(1000);
      continue;
    }
    buffer = reinterpret_cast<void*>(reinterpret_cast<uint64_t>(tx_msg_buffer) +
                                     index * sizeof(VineyardMsg));
    return Status::OK();
  }
}

Status RDMAServer::GetRXFreeMsgBuffer(void*& buffer) {
  if (state == STOPED) {
    return Status::Invalid("Server is stoped.");
  }
  while (true) {
    std::lock_guard<std::mutex> lock(rx_msg_buffer_mutex_);
    int index = FindEmptySlot(rx_buffer_bitmaps, rx_bitmap_num);
    if (index == -1) {
      usleep(1000);
      continue;
    }
    buffer = reinterpret_cast<void*>(reinterpret_cast<uint64_t>(rx_msg_buffer) +
                                     index * sizeof(VineyardMsg));
    return Status::OK();
  }
}

Status RDMAServer::ReleaseRXBuffer(void* buffer) {
  if (state == STOPED) {
    return Status::Invalid("Server is stoped.");
  }
  if ((uint64_t) buffer < (uint64_t) rx_msg_buffer ||
      (uint64_t) buffer >= (uint64_t) rx_msg_buffer + (uint64_t) rx_msg_size) {
    return Status::Invalid("Invalid buffer address.");
  }
  if (((uint64_t) buffer - (uint64_t) rx_msg_buffer) % sizeof(VineyardMsg) !=
      0) {
    return Status::Invalid("Invalid buffer address.");
  }
  std::lock_guard<std::mutex> lock(rx_msg_buffer_mutex_);
  int index =
      ((uint64_t) buffer - (uint64_t) rx_msg_buffer) / sizeof(VineyardMsg);
  rx_buffer_bitmaps[index / 64] |= 1 << (index % 64);
  return Status::OK();
}

Status RDMAServer::ReleaseTXBuffer(void* buffer) {
  if (state == STOPED) {
    return Status::Invalid("Server is stoped.");
  }
  if ((uint64_t) buffer < (uint64_t) tx_msg_buffer ||
      (uint64_t) buffer >= (uint64_t) tx_msg_buffer + (uint64_t) tx_msg_size) {
    return Status::Invalid("Invalid buffer address.");
  }
  if (((uint64_t) buffer - (uint64_t) tx_msg_buffer) % sizeof(VineyardMsg) !=
      0) {
    return Status::Invalid("Invalid buffer address.");
  }
  std::lock_guard<std::mutex> lock(tx_msg_buffer_mutex_);
  int index =
      ((uint64_t) buffer - (uint64_t) tx_msg_buffer) / sizeof(VineyardMsg);
  tx_buffer_bitmaps[index / 64] |= 1 << (index % 64);
  return Status::OK();
}

Status RDMAServer::GetRXCompletion(int timeout, void** context) {
  while (true) {
    int ret = this->GetCompletion(rxcq, timeout == -1 ? 500 : timeout, context);
    if (ret == -FI_ETIMEDOUT) {
      if (timeout > 0) {
        return Status::Invalid("GetRXCompletion timeout");
      } else {
        if (state == STOPED) {
          return Status::Invalid("GetRXCompletion stopped");
        }
        continue;
      }
    } else if (ret < 0) {
      if (ret == -FI_ECANCELED) {
        // client crashed
        return Status::ConnectionError("Client crashed.");
      } else {
        return Status::Invalid(fi_strerror(-ret));
      }
    } else {
      return Status::OK();
    }
  }
}

Status RDMAServer::GetTXCompletion(int timeout, void** context) {
  while (true) {
    int ret = this->GetCompletion(txcq, timeout == -1 ? 500 : timeout, context);
    if (ret == -FI_ETIMEDOUT) {
      if (timeout > 0) {
        return Status::Invalid("GetTXCompletion timeout");
      } else {
        if (state == STOPED) {
          return Status::Invalid("GetTXCompletion stopped");
        }
        continue;
      }
    } else if (ret < 0) {
      if (ret == -FI_ECANCELED) {
        // client crashed
        return Status::ConnectionError("Client crashed.");
      } else {
        return Status::Invalid(fi_strerror(-ret));
      }
    } else {
      return Status::OK();
    }
  }
}

Status RDMAServer::CloseConnection(uint64_t rdma_conn_id) {
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  if (ep_map_.find(rdma_conn_id) == ep_map_.end()) {
    return Status::Invalid("Failed to find buffer context.");
  }
  fid_ep* ep = ep_map_[rdma_conn_id];
  ep_map_.erase(rdma_conn_id);
  return CloseResource(ep, "client endpoint");
}

size_t RDMAServer::GetServerMaxRegisterSize(void* addr, size_t min_size,
                                            size_t max_size) {
  return IRDMA::GetMaxRegisterSizeImpl(addr, min_size, max_size, domain);
}

bool RDMAServer::IsStopped() { return (state == STOPED); }

Status RDMAServer::Stop() {
  state = STOPED;
  return Status::OK();
}
#else

Status RDMAServer::Make(std::shared_ptr<RDMAServer>& ptr, int port) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::Send(uint64_t clientID, void* buf, size_t size, void* ctx) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::Recv(uint64_t clientID, void* buf, size_t size, void* ctx) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::Send(void* ep, void* buf, size_t size, void* ctx) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::Recv(void* ep, void* buf, size_t size, void* ctx) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::Read(uint64_t clientID, void* buf, size_t size,
                        uint64_t remote_address, uint64_t rkey, void* mr_desc,
                        void* ctx) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::Write(uint64_t clientID, void* buf, size_t size,
                         uint64_t remote_address, uint64_t rkey, void* mr_desc,
                         void* ctx) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::GetTXFreeMsgBuffer(void*& buffer) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::GetRXFreeMsgBuffer(void*& buffer) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::GetRXCompletion(int timeout, void** context) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::GetTXCompletion(int timeout, void** context) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::RegisterMemory(RegisterMemInfo& memInfo) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::DeregisterMemory(RegisterMemInfo& memInfo) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::Close() { return Status::OK(); }

Status RDMAServer::Stop() { return Status::OK(); }

Status RDMAServer::AddClient(uint64_t& rdma_conn_id, void* ep) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::CloseConnection(uint64_t rdma_conn_id) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

bool RDMAServer::IsStopped() { return true; }

Status RDMAServer::ReleaseRXBuffer(void* buffer) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::ReleaseTXBuffer(void* buffer) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

size_t RDMAServer::GetServerMaxRegisterSize(void* addr, size_t min_size,
                                            size_t max_size) {
  return 0;
}

Status RDMAServer::GetEvent(VineyardEventEntry& event) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::PrepareConnection(VineyardEventEntry event) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

Status RDMAServer::FinishConnection(uint64_t& rdma_conn_id,
                                    VineyardEventEntry event) {
  return Status::Invalid("RDMA is not supported on this platform.");
}

#endif  // defined(__linux__)

}  // namespace vineyard
