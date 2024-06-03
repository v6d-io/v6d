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

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>

#include <memory>
#include <string>

#include "common/util/logging.h"
#include "common/util/status.h"

#include "common/rdma/rdma_server.h"
#include "common/rdma/util.h"

namespace vineyard {

Status RDMAServer::Make(std::shared_ptr<RDMAServer>& ptr, int port) {
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
  // hints->fabric_attr = (fi_fabric_attr*) malloc(sizeof
  // *(hints->fabric_attr));
  hints->fabric_attr = new fi_fabric_attr;
  memset(hints->fabric_attr, 0, sizeof *(hints->fabric_attr));
  hints->fabric_attr->prov_name = strdup("verbs");

  return Make(ptr, hints, port);
}

Status RDMAServer::Make(std::shared_ptr<RDMAServer>& ptr, fi_info* hints,
                        int port) {
  if (!hints) {
    LOG(ERROR) << "Invalid fabric hints info.";
    return Status::Invalid("Invalid fabric hints info.");
  }

  LOG(INFO) << "Make RDMAServer";
  ptr = std::make_shared<RDMAServer>();

  uint64_t flags = 0;
  CHECK_ERROR(
      !fi_getinfo(VINEYARD_FIVERSION, NULL, std::to_string(port).c_str(), flags,
                  hints, &(ptr->fi)),
      "fi_getinfo failed.");

  CHECK_ERROR(!fi_fabric(ptr->fi->fabric_attr, &ptr->fabric, NULL),
              "fi_fabric failed.");

  ptr->eq_attr.wait_obj = FI_WAIT_UNSPEC;
  CHECK_ERROR(!fi_eq_open(ptr->fabric, &ptr->eq_attr, &ptr->eq, NULL),
              "fi_eq_open failed.");

  CHECK_ERROR(!fi_passive_ep(ptr->fabric, ptr->fi, &ptr->pep, NULL),
              "fi_passive_ep failed.");

  CHECK_ERROR(!fi_pep_bind(ptr->pep, &ptr->eq->fid, 0), "fi_pep_bind failed.");

  CHECK_ERROR(!fi_domain(ptr->fabric, ptr->fi, &ptr->domain, NULL),
              "fi_domain failed.");

  memset(&ptr->cq_attr, 0, sizeof cq_attr);
  ptr->cq_attr.format = FI_CQ_FORMAT_CONTEXT;
  ptr->cq_attr.wait_obj = FI_WAIT_NONE;
  ptr->cq_attr.wait_cond = FI_CQ_COND_NONE;
  ptr->cq_attr.size = ptr->fi->rx_attr->size;
  CHECK_ERROR(!fi_cq_open(ptr->domain, &ptr->cq_attr, &ptr->rxcq, NULL),
              "fi_cq_open failed.");

  ptr->cq_attr.size = ptr->fi->tx_attr->size;
  CHECK_ERROR(!fi_cq_open(ptr->domain, &ptr->cq_attr, &ptr->txcq, NULL),
              "fi_cq_open failed.");

  // ptr->rx_msg_buffer = malloc(ptr->rx_msg_size);
  ptr->rx_msg_buffer = new char[ptr->rx_msg_size];
  if (!ptr->rx_msg_buffer) {
    return Status::Invalid("Failed to allocate rx buffer.");
  }
  // ptr->tx_msg_buffer = malloc(ptr->rx_msg_size);
  ptr->tx_msg_buffer = new char[ptr->rx_msg_size];
  if (!ptr->tx_msg_buffer) {
    return Status::Invalid("Failed to allocate tx buffer.");
  }

  int bits = ptr->rx_msg_size / sizeof(VineyardMsg);
  ptr->rx_bitmap_num = ptr->tx_bitmap_num = (bits + 63) / 64;
  // ptr->rx_buffer_bitmaps =
  //     (uint64_t*) malloc(ptr->rx_bitmap_num * sizeof(uint64_t));
  ptr->rx_buffer_bitmaps = new uint64_t[ptr->rx_bitmap_num];
  if (!ptr->rx_buffer_bitmaps) {
    return Status::Invalid("Failed to allocate rx buffer bitmap.");
  }
  memset(ptr->rx_buffer_bitmaps, UINT8_MAX,
         ptr->rx_bitmap_num * sizeof(uint64_t));
  // ptr->tx_buffer_bitmaps =
  //     (uint64_t*) malloc(ptr->tx_bitmap_num * sizeof(uint64_t));
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

  LOG(INFO) << "RDMAServer listen....";
  CHECK_ERROR(!fi_listen(ptr->pep), "fi_listen failed.");

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

  RETURN_ON_ERROR(CloseResource(txcq, "transmit comeple queue"));
  RETURN_ON_ERROR(CloseResource(rxcq, "receive comeple queue"));
  RETURN_ON_ERROR(CloseResource(pep, "passive endpoint"));
  RETURN_ON_ERROR(CloseResource(eq, "event queue"));

  // before closing domain and fabric, we need to close all the resources
  // bound to them, otherwise, the close operation will failed with -FI_EBUSY
  RETURN_ON_ERROR(CloseResource(domain, "domain"));
  RETURN_ON_ERROR(CloseResource(fabric, "fabric"));

  FreeBuffer(rx_msg_buffer);
  FreeBuffer(tx_msg_buffer);

  FreeBuffer(reinterpret_cast<void*&>(rx_buffer_bitmaps));
  FreeBuffer(reinterpret_cast<void*&>(tx_buffer_bitmaps));

  FreeInfo(fi);

  LOG(INFO) << "Free sources from server successfully";
  return Status::OK();
}

Status RDMAServer::WaitConnect(uint64_t& rdma_conn_id) {
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
    CHECK_ERROR(rd == sizeof entry, "fi_eq_sread failed.");
    if (event == FI_SHUTDOWN) {
      fid_ep* closed_ep = container_of(entry.fid, fid_ep, fid);
      LOG(INFO) << "Connection closed.";
      RemoveClient(closed_ep);
      continue;
    }
    break;
  }

  fi_info* client_fi = entry.info;
  CHECK_ERROR(event == FI_CONNREQ, "Unexpected event:" + std::to_string(event));

  if (!entry.info || !entry.info->fabric_attr || !entry.info->domain_attr ||
      !entry.info->ep_attr || !entry.info->tx_attr || !entry.info->rx_attr)
    return Status::Invalid("Invalid fabric info when prepare connection.");

  if (!entry.info->fabric_attr->prov_name || !entry.info->fabric_attr->name ||
      !entry.info->domain_attr->name ||
      entry.info->fabric_attr->api_version != fi->fabric_attr->api_version)
    return Status::Invalid("Invalid fabric info when prepare connection.");

  // prepare new ep
  fid_ep* ep = NULL;
  CHECK_ERROR(!fi_endpoint(domain, client_fi, &ep, NULL),
              "fi_endpoint failed.");

  CHECK_ERROR(!fi_ep_bind(ep, &eq->fid, 0), "fi_ep_bind eq failed.");

  CHECK_ERROR(!fi_ep_bind(ep, &rxcq->fid, FI_RECV), "fi_ep_bind rxcq failed.");

  CHECK_ERROR(!fi_ep_bind(ep, &txcq->fid, FI_SEND), "fi_ep_bind txcq failed.");

  CHECK_ERROR(!fi_enable(ep), "fi_enable failed.");

  CHECK_ERROR(!fi_accept(ep, NULL, 0), "fi_accept failed.");

  CHECK_ERROR(
      fi_eq_sread(eq, &event, &entry, sizeof(entry), -1, 0) == sizeof entry,
      "fi_eq_sread failed.");

  if (event != FI_CONNECTED || entry.fid != &ep->fid) {
    return Status::Invalid("Unexpected event:" + std::to_string(event));
  }

  // rdma_conn_handle = ep;
  AddClient(rdma_conn_id, ep);

  return Status::OK();
}

Status RDMAServer::AddClient(uint64_t& ep_token, void* ep) {
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  ep_token = current_conn_id++;
  ep_map_[ep_token] = reinterpret_cast<fid_ep*>(ep);
  return Status::OK();
}

Status RDMAServer::RemoveClient(uint64_t ep_token) {
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  ep_map_.erase(ep_token);
  return Status::OK();
}

Status RDMAServer::RemoveClient(fid_ep* ep) {
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  for (auto iter = ep_map_.begin(); iter != ep_map_.end(); iter++) {
    if (iter->second == ep) {
      ep_map_.erase(iter);
      return Status::OK();
    }
  }
  return Status::OK();
}

Status RDMAServer::RegisterMemory(RegisterMemInfo& memInfo) {
  fid_mr* new_mr = NULL;
  RETURN_ON_ERROR(IRDMA::RegisterMemory(
      fi, &new_mr, domain, reinterpret_cast<void*>(memInfo.address),
      memInfo.size, memInfo.rkey, memInfo.mr_desc));
  mr_array.push_back(new_mr);
  return Status::OK();
}

Status RDMAServer::RegisterMemory(fid_mr** mr, void* address, size_t size,
                                  uint64_t& rkey, void*& mr_desc) {
  return IRDMA::RegisterMemory(fi, mr, domain, address, size, rkey, mr_desc);
}

Status RDMAServer::Send(uint64_t ep_token, void* buf, size_t size, void* ctx) {
  LOG(INFO) << "Send";
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  if (ep_map_.find(ep_token) == ep_map_.end()) {
    return Status::Invalid("Failed to find buffer context.");
  }
  fid_ep* ep = ep_map_[ep_token];
  return IRDMA::Send(ep, remote_fi_addr, txcq, buf, size, tx_msg_mr_desc, ctx);
}

Status RDMAServer::Recv(uint64_t ep_token, void* buf, size_t size, void* ctx) {
  LOG(INFO) << "Recv";
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  if (ep_map_.find(ep_token) == ep_map_.end()) {
    return Status::Invalid("Failed to find buffer context.");
  }
  fid_ep* ep = ep_map_[ep_token];
  return IRDMA::Recv(ep, remote_fi_addr, rxcq, buf, size, rx_msg_mr_desc, ctx);
}

Status RDMAServer::Send(void* ep, void* buf, size_t size, void* ctx) {
  return IRDMA::Send(reinterpret_cast<fid_ep*>(ep), remote_fi_addr, txcq, buf,
                     size, tx_msg_mr_desc, ctx);
}

Status RDMAServer::Recv(void* ep, void* buf, size_t size, void* ctx) {
  return IRDMA::Recv(reinterpret_cast<fid_ep*>(ep), remote_fi_addr, rxcq, buf,
                     size, rx_msg_mr_desc, ctx);
}

Status RDMAServer::Read(uint64_t ep_token, void* buf, size_t size,
                        uint64_t remote_address, uint64_t rkey, void* mr_desc,
                        void* ctx) {
  LOG(INFO) << "Read";
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  if (ep_map_.find(ep_token) == ep_map_.end()) {
    return Status::Invalid("Failed to find buffer context.");
  }
  fid_ep* ep = ep_map_[ep_token];
  return IRDMA::Read(ep, remote_fi_addr, rxcq, buf, size, remote_address, rkey,
                     mr_desc, ctx);
}

Status RDMAServer::Write(uint64_t ep_token, void* buf, size_t size,
                         uint64_t remote_address, uint64_t rkey, void* mr_desc,
                         void* ctx) {
  LOG(INFO) << "Write";
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  if (ep_map_.find(ep_token) == ep_map_.end()) {
    return Status::Invalid("Failed to find buffer context.");
  }
  fid_ep* ep = ep_map_[ep_token];
  return IRDMA::Write(ep, remote_fi_addr, txcq, buf, size, remote_address, rkey,
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
    LOG(INFO) << "Get TX index:" << index;
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
    LOG(INFO) << "Get RX index:" << index;
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
  LOG(INFO) << "Release RX index:" << index;
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
  LOG(INFO) << "Release TX index:" << index;
  tx_buffer_bitmaps[index / 64] |= 1 << (index % 64);
  return Status::OK();
}

Status RDMAServer::GetRXCompletion(int timeout, void** context) {
  uint64_t cur = 0;
  while (true) {
    int ret = this->GetCompletion(remote_fi_addr, rxcq, &cur, 1,
                                  timeout == -1 ? 500 : timeout, context);
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
      return Status::Invalid("GetRXCompletion failed");
    } else {
      return Status::OK();
    }
  }
}

Status RDMAServer::GetTXCompletion(int timeout, void** context) {
  uint64_t cur = 0;
  while (true) {
    int ret = this->GetCompletion(remote_fi_addr, txcq, &cur, 1,
                                  timeout == -1 ? 500 : timeout, context);
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
      return Status::Invalid("GetTXCompletion failed");
    } else {
      return Status::OK();
    }
  }
}

Status RDMAServer::CloseConnection(uint64_t rdma_conn_id) {
  LOG(INFO) << "Close connection endpoint!";
  std::lock_guard<std::mutex> lock(ep_map_mutex_);
  if (ep_map_.find(rdma_conn_id) == ep_map_.end()) {
    return Status::Invalid("Failed to find buffer context.");
  }
  fid_ep* ep = ep_map_[rdma_conn_id];
  ep_map_.erase(rdma_conn_id);
  return CloseResource(ep, "client endpoint");
}

bool RDMAServer::IsStopped() { return (state == STOPED); }

}  // namespace vineyard
