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

#include "server/async/rpc_server.h"

#include <memory>
#include <mutex>
#include <utility>

#include "common/util/json.h"
#include "common/util/logging.h"  // IWYU pragma: keep
#include "server/server/vineyard_server.h"

namespace vineyard {

RPCServer::RPCServer(std::shared_ptr<VineyardServer> vs_ptr)
    : SocketServer(vs_ptr),
      rpc_spec_(vs_ptr_->GetSpec()["rpc_spec"]),
      acceptor_(vs_ptr_->GetContext()),
      socket_(vs_ptr_->GetContext()) {
  auto endpoint = getEndpoint(vs_ptr_->GetContext());
  acceptor_.open(endpoint.protocol());
  using reuse_port =
      asio::detail::socket_option::boolean<SOL_SOCKET, SO_REUSEPORT>;
  // reuse address and port for rpc service.
  acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));
  acceptor_.set_option(reuse_port(true));
  acceptor_.bind(endpoint);
  acceptor_.listen();
}

RPCServer::~RPCServer() {
  if (acceptor_.is_open()) {
    acceptor_.close();
  }
  LOG(INFO) << "Close rpc server";
}

Status RPCServer::InitRDMA() {
  if(RDMAServer::Make(this->rdma_server_, DEFAULT_RDMA_PORT).ok()) {
    LOG(INFO) << "Create rdma server successfully";
    this->local_mem_info_.address = (uint64_t)this->vs_ptr_->GetBulkStore()->GetBasePointer();
    this->local_mem_info_.size = this->vs_ptr_->GetBulkStore()->GetBaseSize();
    if(rdma_server_->RegisterMemory(this->local_mem_info_).ok()) {
      LOG(INFO) << "Register rdma memory successfully! Wait port: " << DEFAULT_RDMA_PORT << " connect...";
    } else {
      LOG(INFO) << "Register rdma memory failed! Fall back to TCP.";
      return Status::IOError("Register rdma memory failed");
    }

    doRDMAAccept();
  } else {
    LOG(ERROR) << "Create rdma server failed! Fall back to TCP.";
    return Status::Invalid("Create rdma server failed");
  }
  return Status::OK();
}

void RPCServer::Start() {
  vs_ptr_->RPCReady();
  SocketServer::Start();
  LOG(INFO) << "Vineyard will listen on 0.0.0.0:"
            << rpc_spec_["port"].get<uint32_t>() << " for RPC";
  InitRDMA();
}

asio::ip::tcp::endpoint RPCServer::getEndpoint(asio::io_context&) {
  uint32_t port = rpc_spec_["port"].get<uint32_t>();
  return asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port);
}

Status RPCServer::Register(std::shared_ptr<SocketConnection> conn,
                           const SessionID session_id) {
  if (session_id == RootSessionID()) {
    conn->registered_.store(true);
    return Status::OK();
  }
  std::shared_ptr<VineyardServer> session;
  auto status = this->vs_ptr_->GetRunner()->Get(session_id, session);
  if (status.ok()) {
    conn->switchSession(session);
    conn->registered_.store(true);
  }
  return status;
}

void RPCServer::doAccept() {
  if (!acceptor_.is_open()) {
    return;
  }
  auto self(shared_from_this());
  acceptor_.async_accept(socket_, [self](boost::system::error_code ec) {
    if (!ec) {
      std::lock_guard<std::recursive_mutex> scope_lock(
          self->connections_mutex_);
      if (self->stopped_.load() || self->closable_.load()) {
        return;
      }
      LOG(INFO) << "======Create socket connection=====";
      std::shared_ptr<SocketConnection> conn =
          std::make_shared<SocketConnection>(std::move(self->socket_),
                                             self->vs_ptr_, self,
                                             self->next_conn_id_);
      conn->Start();
      self->connections_.emplace(self->next_conn_id_, conn);
      ++self->next_conn_id_;
    }
    // don't continue when the iocontext being cancelled.
    if (!ec || ec != boost::system::errc::operation_canceled) {
      if (!self->stopped_.load() || !self->closable_.load()) {
        self->doAccept();
      }
    }
  });
}

// Status RPCServer::RDMAExchangeMemInfo() {
//   void *buffer;
//   rdma_server_->GetTXFreeMsgBuffer(buffer);
//   void *remote_msg;
//   rdma_server_->GetRXFreeMsgBuffer(remote_msg);
//   memset(buffer, 0, 64);
//   memset(remote_msg, 0, 64);
//   void *handle;
//   rdma_server_->WaitConnect(handle);
//   LOG(INFO) << "Connected!";
//   rdma_server_->Recv(handle, remote_msg, sizeof(VineyardMsg), nullptr);
//   LOG(INFO) << "Wait";
//   VineyardMsg *msg = (VineyardMsg *)buffer;
//   msg->type = VINEYARD_MSG_EXCHANGE_KEY;
//   msg->remoteMemInfo.remote_address = (uint64_t)local_mem_info_.address;
//   msg->remoteMemInfo.key = local_mem_info_.rkey;
//   msg->remoteMemInfo.len = local_mem_info_.size;
//   rdma_server_->Send(handle, buffer, sizeof(VineyardMsg), nullptr);
//   LOG(INFO) << "Send address:" << local_mem_info_.address;
//   LOG(INFO) << "Send key:" << local_mem_info_.rkey;

//   rdma_server_->GetRXCompletion(-1, nullptr);
//   LOG(INFO) << "Received!";
//   msg = (VineyardMsg *)remote_msg;
//   LOG(INFO) << "Receive remote address: " << msg->remoteMemInfo.remote_address;
//   LOG(INFO) << "Receive remote key: " << msg->remoteMemInfo.key;
//   RegisterMemInfo remote_register_mem_info;
//   remote_register_mem_info.address = msg->remoteMemInfo.remote_address;
//   remote_register_mem_info.rkey = msg->remoteMemInfo.key;
//   remote_register_mem_info.size = msg->remoteMemInfo.len;

//   std::lock_guard<std::recursive_mutex> scope_lock(
//       this->rdma_mutex_);
//   remote_mem_infos_.emplace(msg->remoteMemInfo.rdma_conn_id, remote_register_mem_info);
//   rdma_server_->AddClient(msg->remoteMemInfo.rdma_conn_id, handle);
//   return Status::OK();
// }

// void RPCServer::doRDMAAccept() {
//   this->RDMAExchangeMemInfo();
//   auto self(shared_from_this());
//   boost::asio::post(vs_ptr_->GetContext(), [self]() {
//     self->doRDMAAccept();
//   });
// }

void RPCServer::doRDMAAccept() {
  auto self(shared_from_this());
  boost::asio::post(vs_ptr_->GetContext(), [self]() {
    std::lock_guard<std::recursive_mutex> scope_lock(
        self->rdma_mutex_);
      void *buffer;
      self->rdma_server_->GetTXFreeMsgBuffer(buffer);
      void *remote_msg;
      self->rdma_server_->GetRXFreeMsgBuffer(remote_msg);
      memset(buffer, 0, 64);
      memset(remote_msg, 0, 64);
      void *handle;
      VINEYARD_CHECK_OK(self->rdma_server_->WaitConnect(handle));
      LOG(INFO) << "Connected!";
      self->rdma_server_->Recv(handle, remote_msg, sizeof(VineyardMsg), nullptr);
      LOG(INFO) << "Wait";
      VineyardMsg *msg = (VineyardMsg *)buffer;
      msg->type = VINEYARD_MSG_EXCHANGE_KEY;
      msg->remoteMemInfo.remote_address = (uint64_t)self->local_mem_info_.address;
      msg->remoteMemInfo.key = self->local_mem_info_.rkey;
      msg->remoteMemInfo.len = self->local_mem_info_.size;
      self->rdma_server_->Send(handle, buffer, sizeof(VineyardMsg), nullptr);
      LOG(INFO) << "Send address:" << self->local_mem_info_.address;
      LOG(INFO) << "Send key:" << self->local_mem_info_.rkey;

      self->rdma_server_->GetRXCompletion(-1, nullptr);
      LOG(INFO) << "Received!";
      msg = (VineyardMsg *)remote_msg;
      LOG(INFO) << "Receive remote address: " << msg->remoteMemInfo.remote_address;
      LOG(INFO) << "Receive remote key: " << msg->remoteMemInfo.key;
      LOG(INFO) << "rdma conn id:" << msg->remoteMemInfo.rdma_conn_id;
      self->rdma_server_->AddClient(msg->remoteMemInfo.rdma_conn_id, handle);

      boost::asio::post(self->vs_ptr_->GetContext(), [self]() {
        self->doRDMAAccept();
      });
  });
}

}  // namespace vineyard
