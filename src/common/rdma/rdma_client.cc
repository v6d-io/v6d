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
#include "rdma_client.h"

namespace vineyard {

Status RDMAClient::Make(std::shared_ptr<RDMAClient> &ptr, std::string server_address, int port) {
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

  return Make(ptr, hints, server_address, port);
}

Status RDMAClient::Make(std::shared_ptr<RDMAClient> &ptr, fi_info *hints, std::string server_address, int port) {
  if(!hints) {
    LOG(ERROR) << "Invalid fabric hints info";
    return Status::Invalid("Invalid fabric hints info");
  }

  ptr = std::make_shared<RDMAClient>();

  uint64_t flags = 0;
  CHECK_ERROR(!fi_getinfo(VINEYARD_FIVERSION, server_address.c_str(), std::to_string(port).c_str(), flags, hints, &(ptr->fi)), "fi_getinfo failed\n")

  CHECK_ERROR(!fi_fabric(ptr->fi->fabric_attr, &ptr->fabric, NULL), "fi_fabric failed\n");

  ptr->eq_attr.wait_obj = FI_WAIT_UNSPEC;
  CHECK_ERROR(!fi_eq_open(ptr->fabric, &ptr->eq_attr, &ptr->eq, NULL), "fi_eq_open failed\n");

  CHECK_ERROR(!fi_domain(ptr->fabric, ptr->fi, &ptr->domain, NULL), "fi_domain failed\n");

  memset(&ptr->cq_attr, 0, sizeof cq_attr);
  ptr->cq_attr.format = FI_CQ_FORMAT_CONTEXT;
  ptr->cq_attr.wait_obj = FI_WAIT_NONE;
  ptr->cq_attr.wait_cond = FI_CQ_COND_NONE;
  ptr->cq_attr.size = ptr->fi->rx_attr->size;
  CHECK_ERROR(!fi_cq_open(ptr->domain, &ptr->cq_attr, &ptr->rxcq, NULL), "fi_cq_open failed\n");

  ptr->cq_attr.size = ptr->fi->tx_attr->size;
  CHECK_ERROR(!fi_cq_open(ptr->domain, &ptr->cq_attr, &ptr->txcq, NULL), "fi_cq_open failed\n");

  CHECK_ERROR(!fi_endpoint(ptr->domain, ptr->fi, &ptr->ep, NULL), "fi_endpoint failed\n");

  CHECK_ERROR(!fi_ep_bind(ptr->ep, &ptr->eq->fid, 0), "fi_ep_bind eq failed\n");

  CHECK_ERROR(!fi_ep_bind(ptr->ep, &ptr->rxcq->fid, FI_RECV), "fi_ep_bind rxcq failed\n");

  CHECK_ERROR(!fi_ep_bind(ptr->ep, &ptr->txcq->fid, FI_SEND), "fi_ep_bind txcq failed\n");

  CHECK_ERROR(!fi_enable(ptr->ep), "fi_enable failed\n");

  LOG(INFO) << "size: " << ptr->fi->rx_attr->size << " " << ptr->fi->tx_attr->size << "\n";
  ptr->rx_msg_buffer = malloc(ptr->fi->rx_attr->size);
  if (!ptr->rx_msg_buffer) {
    return Status::Invalid("Failed to allocate rx buffer\n");
  }
  ptr->tx_msg_buffer = malloc(ptr->fi->tx_attr->size);
  if (!ptr->tx_msg_buffer) {
    return Status::Invalid("Failed to allocate tx buffer\n");
  }

  ptr->RegisterMemory(ptr->rx_msg_buffer, ptr->rx_msg_size, ptr->rx_msg_key, ptr->rx_msg_mr_desc);
  ptr->RegisterMemory(ptr->tx_msg_buffer, ptr->tx_msg_size, ptr->tx_msg_key, ptr->tx_msg_mr_desc);

  return Status::OK();
}

Status RDMAClient::Connect() {
  CHECK_ERROR(!fi_connect(ep, fi->dest_addr, NULL, 0), "fi_connect failed\n");

	fi_eq_cm_entry entry;
	uint32_t event;

	CHECK_ERROR(fi_eq_sread(eq, &event, &entry, sizeof(entry), -1, 0) == sizeof(entry), "fi_eq_sread failed\n");

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
    return Status::Invalid("Unexpected event" + std::to_string(event));
	}

  return Status::OK();
}

Status RDMAClient::GetRXCompletion(int timeout, void **context) {
  LOG(INFO) << "GetRXCompletion";
  uint64_t cur = 0;
  return this->GetCompletion(remote_fi_addr, rxcq, &cur, 1, timeout, context);
}

Status RDMAClient::GetTXCompletion(int timeout, void **context) {
  // TBD
  uint64_t cur = 0;
  LOG(INFO) << "GetTXCompletion";
  return this->GetCompletion(remote_fi_addr, txcq, &cur, 1, timeout, context);
}

Status RDMAClient::SendMemInfoToServer(void *buffer, uint64_t size) {
  Send(buffer, size, NULL);
  return Status::OK();
}

Status RDMAClient::GetTXFreeMsgBuffer(void *&buffer) {
  // TBD
  buffer = tx_msg_buffer;
  return Status::OK();
}

Status RDMAClient::GetRXFreeMsgBuffer(void *&buffer) {
  // TBD
  buffer = rx_msg_buffer;
  return Status::OK();
}

Status RDMAClient::RegisterMemory(RegisterMemInfo &memInfo) {
  VINEYARD_CHECK_OK(IRDMA::RegisterMemory(fi, &mr, domain, (void *)memInfo.address, memInfo.size, memInfo.rkey, memInfo.mr_desc));
  return Status::OK();
}

Status RDMAClient::RegisterMemory(void *address, size_t size, uint64_t &rkey, void* &mr_desc) {
  VINEYARD_CHECK_OK(IRDMA::RegisterMemory(fi, &mr, domain, address, size, rkey, mr_desc));
  return Status::OK();
}

Status RDMAClient::Send(void *buf, size_t size, void *ctx) {
  LOG(INFO) << "Send";
  return IRDMA::Send(ep, remote_fi_addr, txcq, buf, size, tx_msg_mr_desc, ctx);
}

Status RDMAClient::Recv(void *buf, size_t size, void *ctx) {
  LOG(INFO) << "Recv";
  return IRDMA::Recv(ep, remote_fi_addr, rxcq, buf, size, rx_msg_mr_desc, ctx);
}

Status RDMAClient::Read(void *buf, size_t size, uint64_t remote_address, uint64_t key, void* mr_desc, void *ctx) {
  LOG(INFO) << "Read";
  return IRDMA::Read(ep, remote_fi_addr, rxcq, buf, size, remote_address, key, mr_desc, ctx);
}

Status RDMAClient::Write(void *buf, size_t size, uint64_t remote_address, uint64_t key, void* mr_desc, void *ctx) {
  LOG(INFO) << "Write";
  return IRDMA::Write(ep, remote_fi_addr, txcq, buf, size, remote_address, key, mr_desc, ctx);
}

Status RDMAClient::Close() {
  // TBD: notifiy the server to close the connection
  return Release();
}

}  // namespace vineyard
