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

#include "server/async/netlink_server.h"

#include <mutex>
#include <string>
#include <utility>
#include <thread>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

#include "common/util/json.h"
#include "common/util/logging.h"
#include "server/server/vineyard_server.h"
#include "common/util/protocols.h"

#define PAGE_SIZE (0x1000)
#define PAGE_DOWN(x) (x & (~(PAGE_SIZE - 1)))
namespace vineyard {

void PrintJsonElement(const json &tree)
{
  LOG(INFO) << __func__;
  for (auto iter = tree.begin(); iter != tree.end(); iter++) {
    LOG(INFO) << *iter;
  }
}

NetLinkServer::NetLinkServer(std::shared_ptr<VineyardServer> vs_ptr)
    : SocketServer(vs_ptr),
      nlh(nullptr),
      req_mem(nullptr),
      result_mem(nullptr),
      obj_info_mem(nullptr),
      base_object_id(std::numeric_limits<uintptr_t>::max()) {
  LOG(INFO) << __func__;
}

NetLinkServer::~NetLinkServer() {
  LOG(INFO) << __func__;
  free(nlh);
}

void NetLinkServer::InitNetLink() {
  socket_fd = socket(AF_NETLINK, SOCK_RAW, NETLINK_VINEYARD);
  if (socket_fd < 0) {
    LOG(INFO) << "NetLinkServer start error";
    return;
  }
  memset(&saddr, 0, sizeof(saddr));

  saddr.nl_family = AF_NETLINK;
  saddr.nl_pid = NETLINK_PORT;
  saddr.nl_groups = 0;

  if(bind(socket_fd, (struct sockaddr *)&saddr, sizeof(saddr)) != 0) {
    LOG(INFO) << "NetLinkServer start error";
    return;
  }

  memset(&daddr, 0, sizeof(daddr));
  daddr.nl_family = AF_NETLINK;
  daddr.nl_pid = 0;
  daddr.nl_groups = 0;

  nlh = (struct nlmsghdr *)malloc(NLMSG_SPACE(sizeof(struct vineyard_result_msg)));
  memset(nlh, 0, sizeof(struct nlmsghdr));
  nlh->nlmsg_len = NLMSG_SPACE(sizeof(struct vineyard_result_msg));
  nlh->nlmsg_flags = 0;
  nlh->nlmsg_type = 0;
  nlh->nlmsg_seq = 0;
  nlh->nlmsg_pid = saddr.nl_pid; //self port

  LOG(INFO) << socket_fd << " " << &saddr << " " << &daddr << " " << nlh;
}

void NetLinkServer::InitialBulkField()
{
  std::vector<ObjectID> ids;
  std::vector<std::shared_ptr<Payload>> objects;

  ids.push_back(base_object_id);
  vs_ptr_->GetBulkStore()->GetUnsafe(ids, true, objects);
  if (!objects.empty()) {
    base_pointer = (void *)PAGE_DOWN((uint64_t)(objects[0]->pointer));
    LOG(INFO) << "base_pointer" << base_pointer;
  }
}

void NetLinkServer::Start() {
  LOG(INFO) << __func__;

  InitNetLink();

  InitialBulkField();

  work = new std::thread(thread_routine, this, socket_fd, saddr, daddr, nlh);
  vs_ptr_->NetLinkReady();
}

void NetLinkServer::Close() {
  LOG(INFO) << __func__;
  close(socket_fd);
}

void NetLinkServer::SyncObjectEntryList() {
  LOG(INFO) << __func__;
  // std::unordered_map<ObjectID, json> metas{};
  vineyard_object_info_header *header;

  header = (vineyard_object_info_header *)obj_info_mem;
  if (header) {
    VineyardWriteLock(&header->rw_lock.r_lock, &header->rw_lock.w_lock);
    header->total_file = 0;
    vs_ptr_->ListData(
        "vineyard::Blob", false, std::numeric_limits<size_t>::max(), [this](const Status& status, const json& tree) {
          if(!tree.empty()) {
            PrintJsonElement(tree);
            this->FillFileEntryInfo(tree, OBJECT_TYPE::BLOB);
          }
          return Status::OK();
        });
     VineyardWriteUnlock(&header->rw_lock.w_lock);
  }
}

void NetLinkServer::doAccept() {
  LOG(INFO) << __func__;
}

int NetLinkServer::HandleSet(struct vineyard_kern_user_msg *msg) {
  LOG(INFO) << __func__;
  req_mem = (void *)(msg->request_mem);
  result_mem = (void *)(msg->result_mem);
  obj_info_mem = (void *)(msg->obj_info_mem);
  return 0;
}

int NetLinkServer::HandleOpen() {
  return 0;
}

int NetLinkServer::HandleRead(fopt_param &param) {
  LOG(INFO) << __func__;
  std::vector<ObjectID> ids;
  std::vector<std::shared_ptr<Payload>> objects;
  vineyard_result_msg rmsg;
  void *pointer = NULL;
  uint64_t file_size = 0;

  LOG(INFO) << "id: "<< param.obj_id << " type " << param.type << " offset " << param.offset << " length " << param.length;

  ids.push_back(param.obj_id);
  vs_ptr_->GetData(
      ids, false, false, []() { return true; },
      [](const Status& status, const json& tree) {
        LOG(INFO) << tree;
        return Status::OK();
      });

  vs_ptr_->GetBulkStore()->GetUnsafe(ids, true, objects);
  for (auto iter = objects.begin(); iter != objects.end(); iter++) {
    LOG(INFO) << "pointer:";
    printf("%p\n", (*iter)->pointer);
    pointer = (*iter)->pointer;
    file_size = (*iter)->data_size;
    printf("pointer:%p\n", pointer);
    printf("file size:%lu\n", file_size);
  }

  if (pointer) {
    rmsg.offset = (uint64_t)pointer - (uint64_t)base_pointer;
    rmsg.ret = 0;
    rmsg.size = file_size;
  } else {
    rmsg.ret = -1;
  }
  VineyardSetResultMsg(rmsg);

  return 0;
}

int NetLinkServer::HandleWrite() {
  return 0;
}

int NetLinkServer::HandleCloseOrFsync() {
  SyncObjectEntryList();
  return 0;
}

int NetLinkServer::HandleReadDir() {
  SyncObjectEntryList();
  return 0;
}

int NetLinkServer::HandleFops() {
  vineyard_msg_mem_header *request_header;
  vineyard_request_msg msg;

  request_header = (struct vineyard_msg_mem_header *)req_mem;

  while(VineyardGetRequestMsg(request_header, &msg)) {
    switch(msg.opt) {
    case REQUEST_OPT::OPEN:
      HandleOpen();
      break;
    case REQUEST_OPT::READ:
      HandleRead(msg._fopt_param);
      break;
    case REQUEST_OPT::WRITE:
      HandleWrite();
      break;
    case REQUEST_OPT::CLOSE:
    case REQUEST_OPT::FSYNC:
      HandleCloseOrFsync();
      break;
    }
  }
  return 0;
}

void NetLinkServer::FillFileEntryInfo(const json &tree, enum OBJECT_TYPE type)
{
  LOG(INFO) << __func__;
  vineyard_object_info_header *header;
  vineyard_entry *entrys;
  int i;

  header = (vineyard_object_info_header *)obj_info_mem;
  entrys = (vineyard_entry *)(header + 1);
  i = header->total_file;
  LOG(INFO) << tree;

  if (type == OBJECT_TYPE::BLOB) {
    for (auto iter = tree.begin(); iter != tree.end(); iter++) {
      entrys[i].obj_id = ObjectIDFromString((*iter)["id"].get<std::string>());
      LOG(INFO) << entrys[i].obj_id;
      entrys[i].file_size = (*iter)["length"].get<uint64_t>();
      entrys[i].type = type;
      i++;
    }
  }
  header->total_file = i;
}

bool NetLinkServer::VineyardGetRequestMsg(vineyard_msg_mem_header *header, vineyard_request_msg *msg)
{
  LOG(INFO) << __func__;
  struct vineyard_request_msg *entrys;
  struct vineyard_request_msg *entry = NULL;

  entrys = (struct vineyard_request_msg *)(header + 1);
  VineyardSpinLock(&header->lock);
  LOG(INFO) << header->head_point << " " << header->tail_point;
  if (!MsgEmpty(header->head_point, header->tail_point)) {
    entry = &(entrys[header->tail_point]);
    memcpy(msg, entry, sizeof(*msg));
    header->tail_point++;
    // TODO: ring buffer reset pointer.
  }

  VineyardSpinUnlock(&header->lock);
  if (entry)
    return true;
  return false;
}

int NetLinkServer::VineyardSetResultMsg(vineyard_result_msg &rmsg)
{
  vineyard_result_mem_header *rheader;
  vineyard_result_msg *rmsgs;
  rheader = (vineyard_result_mem_header *)result_mem;
  rmsgs = (vineyard_result_msg *)((uint64_t)result_mem + sizeof(vineyard_result_mem_header));

  VineyardSpinLock(&rheader->lock);
  memcpy(&rmsgs[rheader->head_point], &rmsg, sizeof(vineyard_result_msg));
  rheader->head_point++;
  VineyardSpinUnlock(&rheader->lock);

  return 0;
}

void NetLinkServer::thread_routine(NetLinkServer *ns_ptr, int socket_fd, struct sockaddr_nl saddr, struct sockaddr_nl daddr, struct nlmsghdr *nlh) {
  LOG(INFO) << __func__;
  int ret;
  socklen_t len;
  kmsg kmsg;
  vineyard_result_msg res_msg;

  LOG(INFO) << socket_fd << " " << &saddr << " " << &daddr << " " << nlh;
  memset(&res_msg, 0, sizeof(res_msg));
  res_msg.opt = INIT;

  ns_ptr->SyncObjectEntryList();

  while(1) {
    memcpy(NLMSG_DATA(nlh), &res_msg, sizeof(res_msg));
    ret = sendto(socket_fd, nlh, nlh->nlmsg_len, 0, (struct sockaddr *)&daddr, sizeof(struct sockaddr_nl));
    if (!ret) {
      LOG(INFO) << "NetLinkServer send msg error";
      goto out;
    }

    memset(&kmsg, 0, sizeof(kmsg));
    len = sizeof(struct sockaddr_nl);
    ret = recvfrom(socket_fd, &kmsg, sizeof(kmsg), 0, (struct sockaddr *)&daddr, &len);
    if(!ret)
    {
        LOG(INFO) << "Recv form kernel error\n";
        goto out;
    }

    LOG(INFO) << "Receive from kernel";

    switch(kmsg.msg.opt) {
    case USER_KERN_OPT::SET:
      res_msg.ret = ns_ptr->HandleSet(&kmsg.msg);
      res_msg.opt = WAIT;
      break;
    case USER_KERN_OPT::FOPT:
      res_msg.opt = FOPT;
      res_msg.ret = ns_ptr->HandleFops();
      break;
    case USER_KERN_OPT::EXIT:
      LOG(INFO) << "Bye! Handler thread exit!";
      goto out;
    default:
      LOG(INFO) << "Error req";
    }
  }

out:
  return;
}

}