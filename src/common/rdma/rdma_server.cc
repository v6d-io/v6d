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
#include <string>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>

#include "common/util/logging.h"
#include "common/util/status.h"

#include "util.h"
#include "rdma_server.h"

namespace vineyard {

Status RDMAServer::Make(std::shared_ptr<RDMAServer> &ptr, int port) {
  fi_info *hints = fi_allocinfo();
  if (!hints) {
    return Status::Invalid("Failed to allocate fabric info");
  }

  hints->caps = FI_MSG | FI_RMA | FI_WRITE | FI_REMOTE_WRITE | FI_READ | FI_REMOTE_READ;
  hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
  hints->mode = FI_CONTEXT;
  hints->domain_attr->threading = FI_THREAD_DOMAIN;
  hints->addr_format = FI_FORMAT_UNSPEC;
  hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_ENDPOINT |
                                FI_MR_ALLOCATED | FI_MR_PROV_KEY |
                                FI_MR_VIRT_ADDR | FI_MR_RAW;
  hints->tx_attr->tclass = FI_TC_BULK_DATA;
  hints->ep_attr->type = FI_EP_MSG;
  hints->fabric_attr = (fi_fabric_attr *)malloc(sizeof *(hints->fabric_attr));
  memset(hints->fabric_attr, 0, sizeof *(hints->fabric_attr));
  hints->fabric_attr->prov_name = strdup("verbs");

  return Make(ptr, hints, port);
}

Status RDMAServer::Make(std::shared_ptr<RDMAServer> &ptr, fi_info *hints, int port) {
  if(!hints) {
    LOG(ERROR) << "Invalid fabric hints info";
    return Status::Invalid("Invalid fabric hints info");
  }

  LOG(INFO) << "Make RDMAServer\n";
  ptr = std::make_shared<RDMAServer>();

  uint64_t flags = 0;
  LOG(INFO) << "fi_getinfo\n";
  CHECK_ERROR(!fi_getinfo(VINEYARD_FIVERSION, NULL, std::to_string(port).c_str(), flags, hints, &(ptr->fi)), "fi_getinfo failed\n");

  CHECK_ERROR(!fi_fabric(ptr->fi->fabric_attr, &ptr->fabric, NULL), "fi_fabric failed\n");

  ptr->eq_attr.wait_obj = FI_WAIT_UNSPEC;
  CHECK_ERROR(!fi_eq_open(ptr->fabric, &ptr->eq_attr, &ptr->eq, NULL), "fi_eq_open failed\n");

  CHECK_ERROR(!fi_passive_ep(ptr->fabric, ptr->fi, &ptr->pep, NULL), "fi_passive_ep failed\n");

  CHECK_ERROR(!fi_pep_bind(ptr->pep, &ptr->eq->fid, 0), "fi_pep_bind failed\n");

  CHECK_ERROR(!fi_domain(ptr->fabric, ptr->fi, &ptr->domain, NULL), "fi_domain failed\n");

  memset(&ptr->cq_attr, 0, sizeof cq_attr);
  ptr->cq_attr.format = FI_CQ_FORMAT_CONTEXT;
  ptr->cq_attr.wait_obj = FI_WAIT_NONE;
  ptr->cq_attr.wait_cond = FI_CQ_COND_NONE;
  ptr->cq_attr.size = ptr->fi->rx_attr->size;
  CHECK_ERROR(!fi_cq_open(ptr->domain, &ptr->cq_attr, &ptr->rxcq, NULL), "fi_cq_open failed\n");

  ptr->cq_attr.size = ptr->fi->tx_attr->size;
  CHECK_ERROR(!fi_cq_open(ptr->domain, &ptr->cq_attr, &ptr->txcq, NULL), "fi_cq_open failed\n");

  ptr->rx_msg_buffer = malloc(ptr->rx_msg_size);
  if (!ptr->rx_msg_buffer) {
    return Status::Invalid("Failed to allocate rx buffer\n");
  }
  ptr->tx_msg_buffer = malloc(ptr->rx_msg_size);
  if (!ptr->tx_msg_buffer) {
    return Status::Invalid("Failed to allocate tx buffer\n");
  }

  ptr->RegisterMemory(ptr->rx_msg_buffer, ptr->rx_msg_size, ptr->rx_msg_key, ptr->rx_msg_mr_desc);
  ptr->RegisterMemory(ptr->tx_msg_buffer, ptr->tx_msg_size, ptr->tx_msg_key, ptr->tx_msg_mr_desc);

  ptr->port = port;

  return Status::OK();
}

Status RDMAServer::Close() {

  return Release();
}

Status RDMAServer::WaitConnect() {
  printf("fi_listen\n");
  CHECK_ERROR(!fi_listen(pep), "fi_listen failed\n");

	struct fi_eq_cm_entry entry;
	uint32_t event;

  CHECK_ERROR(fi_eq_sread(eq, &event, &entry, sizeof entry, -1, 0) == sizeof entry, "fi_eq_sread failed\n");

  fi_info *client_fi = entry.info;
  CHECK_ERROR(event == FI_CONNREQ, "Unexpected event\n");

	if (!entry.info || !entry.info->fabric_attr || !entry.info->domain_attr ||
	    !entry.info->ep_attr || !entry.info->tx_attr || !entry.info->rx_attr)
		return Status::Invalid("Invalid fabric info when prepare connection");

	if (!entry.info->fabric_attr->prov_name ||
	    !entry.info->fabric_attr->name || !entry.info->domain_attr->name ||
	    entry.info->fabric_attr->api_version != fi->fabric_attr->api_version)
		return Status::Invalid("Invalid fabric info when prepare connection");

  // prepare new ep

  CHECK_ERROR(!fi_endpoint(domain, client_fi, &ep, NULL), "fi_endpoint failed\n");

  CHECK_ERROR(!fi_ep_bind(ep, &eq->fid, 0), "fi_ep_bind eq failed\n");

  CHECK_ERROR(!fi_ep_bind(ep, &rxcq->fid, FI_RECV), "fi_ep_bind rxcq failed\n");

  CHECK_ERROR(!fi_ep_bind(ep, &txcq->fid, FI_SEND), "fi_ep_bind txcq failed\n");

  CHECK_ERROR(!fi_enable(ep), "fi_enable failed\n");

  CHECK_ERROR(!fi_accept(ep, NULL, 0), "fi_accept failed\n");

  LOG(INFO) << "accept done?\n";
	CHECK_ERROR(fi_eq_sread(eq, &event, &entry, sizeof(entry), -1, 0) == sizeof entry, "fi_eq_sread failed\n");

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
    return Status::Invalid("Unexpected event\n");
  }

  AddClient(TEST_CLIENT_ID, ep);

  // VINEYARD_CHECK_OK(Recv(buf, 1024, bufferContext.mr_desc, NULL));

  // TODO: let worker to check complete queue
  // TODO: get client instance id
  // TODO: set buffer context

  return Status::OK();
}

Status RDMAServer::AddClient(uint64_t clientID, fid_ep *ep) {
  VineyardBufferContext bufferContext;
  bufferContext.ep = ep;
  bufferContext.rkey = mem_key;
  buffer_map[clientID] = bufferContext;
  return Status::OK();
}

Status RDMAServer::RegisterMemory(RegisterMemInfo &memInfo) {
  RETURN_ON_ERROR(IRDMA::RegisterMemory(fi, &mr, domain, memInfo.address, memInfo.size , memInfo.rkey, memInfo.mr_desc));
  mem_key = memInfo.rkey;
  return Status::OK();
}

Status RDMAServer::RegisterMemory(void *address, size_t size, uint64_t &rkey, void* &mr_desc) {
  return IRDMA::RegisterMemory(fi, &mr, domain, address, size , rkey, mr_desc);
}

Status RDMAServer::Send(uint64_t clientID, void* buf, size_t size, void* ctx) {
  if (buffer_map.find(clientID) == buffer_map.end()) {
    return Status::Invalid("Failed to find buffer context");
  }
  LOG(INFO) << "Send";
  VineyardBufferContext bufferContext = buffer_map[clientID];
  return IRDMA::Send(bufferContext.ep, remote_fi_addr, txcq, buf, size, tx_msg_mr_desc, ctx);
}

Status RDMAServer::Recv(uint64_t clientID, void* buf, size_t size, void* ctx) {
  if (buffer_map.find(clientID) == buffer_map.end()) {
    return Status::Invalid("Failed to find buffer context");
  }
  LOG(INFO) << "Recv";
  VineyardBufferContext bufferContext = buffer_map[clientID];
  return IRDMA::Recv(bufferContext.ep, remote_fi_addr, rxcq, buf, size, rx_msg_mr_desc, ctx);
}

Status RDMAServer::Read(uint64_t clientID, void *buf, size_t size, uint64_t remote_address, uint64_t rkey, void *mr_desc, void* ctx) {
  if (buffer_map.find(clientID) == buffer_map.end()) {
    return Status::Invalid("Failed to find buffer context");
  }
  LOG(INFO) << "Read";
  VineyardBufferContext bufferContext = buffer_map[clientID];
  return IRDMA::Read(bufferContext.ep, remote_fi_addr, rxcq, buf, size, remote_address, rkey, data_mem_desc, ctx);
}

Status RDMAServer::Write(uint64_t clientID, void *buf, size_t size, uint64_t remote_address, uint64_t rkey, void *mr_desc, void* ctx) {
  if (buffer_map.find(clientID) == buffer_map.end()) {
    return Status::Invalid("Failed to find buffer context");
  }
  LOG(INFO) << "Write";
  VineyardBufferContext bufferContext = buffer_map[clientID];
  return IRDMA::Write(bufferContext.ep, remote_fi_addr, txcq, buf, size, remote_address, rkey, data_mem_desc, ctx);
}

Status RDMAServer::GetTXFreeMsgBuffer(void *&buffer) {
  // TBD
  buffer = tx_msg_buffer;
  return Status::OK();
}

Status RDMAServer::GetRXFreeMsgBuffer(void *&buffer) {
  // TBD
  buffer = rx_msg_buffer;
  return Status::OK();
}

Status RDMAServer::GetRXCompletion(int timeout, void **context) {
  uint64_t cur = 0;
  return this->GetCompletion(ep, remote_fi_addr, rxcq, &cur, 1, timeout, context);
}

Status RDMAServer::GetTXCompletion(int timeout, void **context) {
  // TBD
  uint64_t cur = 0;
  return this->GetCompletion(ep, remote_fi_addr, txcq, &cur, 1, timeout, context);
}

}  // namespace vineyard
