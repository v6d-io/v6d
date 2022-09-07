/** Copyright 2020-2022 Alibaba Group Holding Limited.

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
#if defined(BUILD_NETLINK_SERVER) && BUILD_NETLINK_SERVER
#ifdef __linux__
#include "server/async/netlink_server.h"

#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

#include "common/util/json.h"
#include "common/util/logging.h"
#include "common/util/protocols.h"
#include "server/server/vineyard_server.h"

#define OFF (0)
#define ON (1)
#define DEBUG OFF
#define PAGE_SIZE (0x1000)
#define PAGE_DOWN(x) ((x) & (~(PAGE_SIZE - 1)))
namespace vineyard {

// Debug tools
static void PrintJsonElement(const json& tree) {
#if DEBUG == ON
  LOG(INFO) << __func__;
  for (auto iter = tree.begin(); iter != tree.end(); iter++) {
    LOG(INFO) << *iter;
  }
#endif
}

NetLinkServer::NetLinkServer(std::shared_ptr<VineyardServer> vs_ptr)
    : SocketServer(vs_ptr),
      nlh(nullptr),
      obj_info_mem(nullptr),
      base_object_id(std::numeric_limits<uintptr_t>::max()) {
  LOG(INFO) << __func__;
}

NetLinkServer::~NetLinkServer() {
  LOG(INFO) << __func__;
  close(socket_fd);
  free(nlh);
}

void NetLinkServer::InitNetLink() {
  size_t size;
  socket_fd = socket(AF_NETLINK, SOCK_RAW, NETLINK_VINEYARD);
  if (socket_fd < 0) {
    LOG(INFO) << "If you want to use netlink server, please insert kernel "
                 "module first!";
    return;
  }
  memset(&saddr, 0, sizeof(saddr));

  saddr.nl_family = AF_NETLINK;
  saddr.nl_pid = NETLINK_PORT;
  saddr.nl_groups = 0;

  if (bind(socket_fd, (struct sockaddr*) &saddr, sizeof(saddr)) != 0) {
    LOG(INFO) << "NetLinkServer start error";
    return;
  }

  memset(&daddr, 0, sizeof(daddr));
  daddr.nl_family = AF_NETLINK;
  daddr.nl_pid = 0;
  daddr.nl_groups = 0;

  size = sizeof(vineyard_msg);
  nlh = (struct nlmsghdr*) malloc(NLMSG_SPACE(size));
  memset(nlh, 0, sizeof(struct nlmsghdr));
  nlh->nlmsg_len = NLMSG_SPACE(size);
  nlh->nlmsg_flags = 0;
  nlh->nlmsg_type = 0;
  nlh->nlmsg_seq = 0;
  nlh->nlmsg_pid = saddr.nl_pid;
}

uint64_t NetLinkServer::GetServerBulkField() {
  // FIXME: it is temp implement.
  std::vector<ObjectID> ids;
  std::vector<std::shared_ptr<Payload>> objects;

  ids.push_back(base_object_id);
  vs_ptr_->GetBulkStore()->GetUnsafe(ids, true, objects);
  if (!objects.empty())
    return (uint64_t)(objects[0]->pointer);
  return 0;
}

uint64_t NetLinkServer::GetServerBulkSize() {
  std::vector<ObjectID> ids;
  std::vector<std::shared_ptr<Payload>> objects;

  ids.push_back(base_object_id);
  vs_ptr_->GetBulkStore()->GetUnsafe(ids, true, objects);
  if (!objects.empty())
    return (uint64_t)(objects[0]->data_size);
  return 0;
}

void NetLinkServer::InitialBulkField() {
  base_pointer = reinterpret_cast<void*>(GetServerBulkField());
}

void NetLinkServer::Start() {
  InitNetLink();
  InitialBulkField();

  work = new std::thread(thread_routine, this, socket_fd, saddr, daddr, nlh);
  vs_ptr_->NetLinkReady();
}

void NetLinkServer::Close() { LOG(INFO) << __func__; }

void NetLinkServer::SyncObjectEntryList() {
  vineyard_object_info_header* header;

  header = reinterpret_cast<vineyard_object_info_header*>(obj_info_mem);
  if (header) {
    VineyardWriteLock(&header->rw_lock.r_lock, &header->rw_lock.w_lock);
    header->total_file = 0;
    vs_ptr_->ListData("vineyard::Blob", false,
                      std::numeric_limits<size_t>::max(),
                      [this](const Status& status, const json& tree) {
                        if (!tree.empty()) {
                          PrintJsonElement(tree);
                          this->FillFileEntryInfo(tree, OBJECT_TYPE::BLOB);
                        }
                        return Status::OK();
                      });
    VineyardWriteUnlock(&header->rw_lock.w_lock);
  }
}

void NetLinkServer::doAccept() { LOG(INFO) << __func__; }

int NetLinkServer::HandleSet(vineyard_request_msg* msg) {
  LOG(INFO) << __func__;
  obj_info_mem = reinterpret_cast<void*>(msg->param._set_param.obj_info_mem);
  SyncObjectEntryList();
  return 0;
}

fopt_ret NetLinkServer::HandleOpen(fopt_param& param) {
  LOG(INFO) << __func__;

  std::vector<ObjectID> ids;
  std::vector<std::shared_ptr<Payload>> objects;
  fopt_ret ret;
  void* pointer = NULL;
  uint64_t file_size = 0;

  ret.ret = -1;
  ids.push_back(param.obj_id);
  vs_ptr_->GetData(
      ids, false, false, []() { return true; },
      [](const Status& status, const json& tree) {
        LOG(INFO) << tree;
        return Status::OK();
      });

  vs_ptr_->GetBulkStore()->GetUnsafe(ids, true, objects);
  for (auto iter = objects.begin(); iter != objects.end(); iter++) {
    pointer = (*iter)->pointer;
    file_size = (*iter)->data_size;
  }

  if (pointer) {
    ret.offset = (uint64_t) pointer - (uint64_t) base_pointer;
    ret.ret = 0;
    ret.size = file_size;
  }

  return ret;
}

fopt_ret NetLinkServer::HandleRead(fopt_param& param) {
  LOG(INFO) << __func__;
  fopt_ret ret;
  ret.ret = 0;

  return ret;
}

fopt_ret NetLinkServer::HandleWrite() {
  fopt_ret ret;
  ret.ret = 0;
  return ret;
}

fopt_ret NetLinkServer::HandleCloseOrFsync() {
  SyncObjectEntryList();
  fopt_ret ret;
  ret.ret = 0;
  return ret;
}

fopt_ret NetLinkServer::HandleReadDir() {
  SyncObjectEntryList();
  fopt_ret ret;
  ret.ret = 0;
  return ret;
}

fopt_ret NetLinkServer::HandleFops(vineyard_request_msg* msg) {
  fopt_ret ret;
  switch (msg->opt) {
  case MSG_OPT::VINEYARD_OPEN:
    return HandleOpen(msg->param._fopt_param);
  case MSG_OPT::VINEYARD_READ:
    return HandleRead(msg->param._fopt_param);
  case MSG_OPT::VINEYARD_WRITE:
    return HandleWrite();
  case MSG_OPT::VINEYARD_CLOSE:
  case MSG_OPT::VINEYARD_FSYNC:
    return HandleCloseOrFsync();
  default:
    LOG(INFO) << "Error opt!";
    return ret;
  }
}

void NetLinkServer::FillFileEntryInfo(const json& tree, enum OBJECT_TYPE type) {
  LOG(INFO) << __func__;
  vineyard_object_info_header* header;
  vineyard_entry* entrys;
  int i = 0;

  header = reinterpret_cast<vineyard_object_info_header*>(obj_info_mem);
  entrys = reinterpret_cast<vineyard_entry*>(header + 1);
  PrintJsonElement(tree);

  if (type == OBJECT_TYPE::BLOB) {
    for (auto iter = tree.begin(); iter != tree.end(); iter++) {
      entrys[i].obj_id = ObjectIDFromString((*iter)["id"].get<std::string>());
      entrys[i].file_size = (*iter)["length"].get<uint64_t>();
      entrys[i].type = type;
      i++;
    }
  }
  header->total_file = i;
}

void NetLinkServer::thread_routine(NetLinkServer* ns_ptr, int socket_fd,
                                   struct sockaddr_nl saddr,
                                   struct sockaddr_nl daddr,
                                   struct nlmsghdr* nlh) {
  LOG(INFO) << __func__;
  int ret;
  socklen_t len;
  kmsg kmsg;
  vineyard_result_msg umsg;
  fopt_ret _fopt_ret;

  if (socket_fd < 0) {
    return;
  }

  memset(&umsg, 0, sizeof(umsg));
  umsg.opt = MSG_OPT::VINEYARD_WAIT;

  while (1) {
    if (!nlh) {
      LOG(INFO) << "If you want to use netlink server, please insert kernel "
                   "module first!";
      goto out;
    }
    memcpy(NLMSG_DATA(nlh), &umsg, sizeof(umsg));
    ret = sendto(socket_fd, nlh, nlh->nlmsg_len, 0, (struct sockaddr*) &daddr,
                 sizeof(struct sockaddr_nl));
    if (!ret) {
      LOG(INFO) << "NetLinkServer send msg error";
      goto out;
    }

    memset(&kmsg, 0, sizeof(kmsg));
    len = sizeof(struct sockaddr_nl);
    ret = recvfrom(socket_fd, &kmsg, sizeof(kmsg), 0, (struct sockaddr*) &daddr,
                   &len);
    if (!ret) {
      LOG(INFO) << "Recv form kernel error\n";
      goto out;
    }

    switch (kmsg.msg.msg.request.opt) {
    case MSG_OPT::VINEYARD_MOUNT:
      ns_ptr->HandleSet(&kmsg.msg.msg.request);
      umsg.ret._set_ret.bulk_addr = ns_ptr->GetServerBulkField();
      umsg.ret._set_ret.bulk_size = ns_ptr->GetServerBulkSize();
      umsg.ret._set_ret.ret = 0;
      umsg.opt = MSG_OPT::VINEYARD_SET_BULK_ADDR;
      break;
    case MSG_OPT::VINEYARD_EXIT:
      LOG(INFO) << "Bye! Handler thread exit!";
      goto out;
    case MSG_OPT::VINEYARD_OPEN:
    case MSG_OPT::VINEYARD_READ:
    case MSG_OPT::VINEYARD_WRITE:
    case MSG_OPT::VINEYARD_CLOSE:
    case MSG_OPT::VINEYARD_FSYNC:
      _fopt_ret = ns_ptr->HandleFops(&kmsg.msg.msg.request);
      memcpy(&umsg.ret._fopt_ret, &_fopt_ret, sizeof(fopt_ret));
      umsg.opt = MSG_OPT::VINEYARD_OPEN;
      break;
    default:
      LOG(INFO) << "Error opt";
    }
  }

out:
  return;
}

}  // namespace vineyard
#endif  // __linux__
#endif  // BUILD_NETLINK_SERVER
