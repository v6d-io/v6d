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

#if defined(BUILD_NETLINK_SERVER) && BUILD_NETLINK_SERVER
#ifdef __linux__
#include <iostream>
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
#define DEBUG ON
#define PAGE_SIZE (0x1000)
#define PAGE_DOWN(x) ((x) & (~(PAGE_SIZE - 1)))

#define MAGIC ("\x93NUMPY")
#define descr           ("'descr': ")
#define fortran_order   ("'fortran_order': ")
#define shape           ("'shape': ")
int8_t majorVersion = 1;
int8_t minorVersion = 0;
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

std::pair<char *, int> ConstructHeader(int type, std::string &tensor_shape, bool order)
{
    char *ret;
    char *temp;
    int16_t header_len;
    int16_t total_len;
    int16_t shape_len;
    char descr_str[17] = { 0 };
    char fortran_order_str[25] = {0};
    char *shape_str;

    // type
    // we now suppose that the platform is little endian
    strcpy(descr_str, descr);
    switch(type) {
    case 1:
        strcpy(descr_str + strlen(descr), "'<i8', ");
        break;
    case 2:
        strcpy(descr_str + strlen(descr), "'<u8', ");
        break;
    default:
        break;
    }

    //fortran
    strcpy(fortran_order_str, fortran_order);
    if(order) {
        strcpy(fortran_order_str + strlen(fortran_order), "True, ");
    } else {
        strcpy(fortran_order_str + strlen(fortran_order), "False, ");
    }


    //shape
    shape_len = tensor_shape.length() + strlen(shape) + 3;
    shape_str = (char *)malloc(sizeof(char) * shape_len);
    memset(shape_str, 0, shape_len);
    strcpy(shape_str, shape);
    strcpy(shape_str + strlen(shape), tensor_shape.c_str());
    shape_str[shape_len - 3] = ',';
    shape_str[shape_len - 2] = ' ';

    total_len = strlen(descr_str) + strlen(fortran_order_str) + strlen(shape_str) + 12;
    total_len = (total_len + 63) & 0xFFC0;
    header_len = total_len - 10;

    ret = (char *)malloc(sizeof(char) * total_len);
    temp = ret;


    memcpy(temp, MAGIC, strlen(MAGIC));
    temp += strlen(MAGIC);
    memcpy(temp, &majorVersion, sizeof(majorVersion));
    temp += sizeof(majorVersion);
    memcpy(temp, &minorVersion, sizeof(minorVersion));
    temp += sizeof(minorVersion);
    memcpy(temp, &header_len, sizeof(header_len));
    temp += sizeof(header_len);

    *temp = '{';
    temp++;
    memcpy(temp, descr_str, strlen(descr_str));
    temp += strlen(descr_str);
    memcpy(temp, fortran_order_str, strlen(fortran_order_str));
    temp += strlen(fortran_order_str);
    memcpy(temp, shape_str, strlen(shape_str));
    temp += strlen(shape_str);
    *temp = '}';
    temp++;

    while (temp - ret < total_len - 1) {
        *temp = ' ';
        temp++;
    }
    *temp = '\n';

    free(shape_str);
    return std::pair<char *, int>(ret, total_len);
}

std::string ConstructTensorShape(std::vector<int> &tensor_shape)
{
  std::string ret;
  ret.push_back('(');
  for (auto iter = tensor_shape.begin(); iter != tensor_shape.end(); iter++) {
    ret.push_back(*iter + '0');
    if (iter + 1 != tensor_shape.end()) {
      ret.push_back(',');
      ret.push_back(' ');
    }
  }
  ret.push_back(')');
  return ret;
}

NetLinkServer::NetLinkServer(std::shared_ptr<VineyardServer> vs_ptr)
    : SocketServer(vs_ptr),
      nlh(nullptr),
      obj_info_mem(nullptr),
      obj_info_lock(0),
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
}

void NetLinkServer::Close() { LOG(INFO) << __func__; }

void NetLinkServer::SyncObjectEntryList() {
  vineyard_object_info_header* header;
  sync_ready = 0;
  LOG(INFO) << __func__;

  header = reinterpret_cast<vineyard_object_info_header*>(obj_info_mem);
  if (header) {
    VineyardWriteLock(&header->rw_lock.r_lock, &header->rw_lock.w_lock);
    header->total_file = 0;
    vs_ptr_->ListData("vineyard::Blob", false,
                      std::numeric_limits<size_t>::max(),
                      [this, header](const Status& status, const json& tree) {
                        LOG(INFO) << "blob";
                        if (!tree.empty()) {
                          PrintJsonElement(tree);
                          this->FillFileEntryInfo(tree, OBJECT_TYPE::BLOB);
                        }
                        this->sync_ready |= _ready::Blob;
                        if (this->sync_ready == _ready::Ready) {
                          LOG(INFO) << "Unlock";
                          VineyardWriteUnlock(&header->rw_lock.w_lock);
                          this->sync_ready = 0;
                        }
                        return Status::OK();
                      });
    vs_ptr_->ListData("vineyard::Tensor*", false,
                      std::numeric_limits<size_t>::max(),
                      [this, header](const Status& status, const json& tree) {
                        LOG(INFO) << "tensor";
                        if (!tree.empty()) {
                          // PrintJsonElement(tree);
                          this->FillFileEntryInfo(tree, OBJECT_TYPE::TENSOR);
                        }
                        this->sync_ready |= _ready::Tensor;
                        if (this->sync_ready == _ready::Ready) {
                          LOG(INFO) << "Unlock";
                          VineyardWriteUnlock(&header->rw_lock.w_lock);
                          this->sync_ready = 0;
                        }
                        return Status::OK();
                      });
  }
}

void NetLinkServer::doAccept() { LOG(INFO) << __func__; }

int NetLinkServer::HandleSet(vineyard_request_msg* msg) {
  LOG(INFO) << __func__;
  obj_info_mem = reinterpret_cast<void*>(msg->param._set_param.obj_info_mem);
  return 0;
}

fopt_ret NetLinkServer::HandleOpen(fopt_param& param) {
  LOG(INFO) << __func__;

  std::vector<ObjectID> ids;
  std::vector<std::shared_ptr<Payload>> objects;
  fopt_ret ret;
  void* pointer = NULL;
  uint64_t file_size = 0;
  unsigned int sync_val = 0;

  ret.ret = -1;
  ids.push_back(param.obj_id);
  LOG(INFO) << param.type;
  // TODO: Use a more elegent way to sync result.
  VineyardSpinLock(&sync_val);
  if (param.type == OBJECT_TYPE::TENSOR) {
    vs_ptr_->GetData(
        ids, false, false, []() { return true; },
        [&sync_val, this, &ret](const Status& status, const json& tree) {
          uint64_t bulk_id;
          if (!tree.empty()) {
            std::vector<ObjectID> ids;
            std::vector<std::shared_ptr<Payload>> objects;
            std::vector<int> tensor_shape;
            std::string tensor_shape_string;
            std::pair<char *, int> header;

            auto iter = tree.begin();
            LOG(INFO) << *iter;
            LOG(INFO) << tree;
            LOG(INFO) << "=========";
            bulk_id = ObjectIDFromString((*iter)["buffer_"]["id"].get<std::string>());
            // tensor_shape = (*iter)["shape_"].get<std::vector<uint64_t>>();
            get_container(*iter, "shape_", tensor_shape);
            for (auto it = tensor_shape.begin(); it != tensor_shape.end(); it++) {
              LOG(INFO) << *it;
            }
            tensor_shape_string = ConstructTensorShape(tensor_shape);
            header = ConstructHeader(2, tensor_shape_string, false);
            for (int i = 0; i < header.second; i++) {
              std::cout << *(header.first + i) << " ";
            }
            std::cout << std::endl;

            LOG(INFO) << bulk_id;
            ids.push_back(bulk_id);
            this->vs_ptr_->GetBulkStore()->GetUnsafe(ids, true, objects);
            //FIXME: maybe there exist more than one blob to store the data.
            for (auto iter = objects.begin(); iter != objects.end(); iter++) {
              LOG(INFO) << "find!";
              LOG(INFO) << (uint64_t)(*iter)->pointer;
              LOG(INFO) << (*iter)->data_size;
              ret.data_offset = (uint64_t)(*iter)->pointer - (uint64_t) base_pointer;
              ret.data_size = (*iter)->data_size;

              // construct npy header
              ObjectID object_id;
              std::shared_ptr<Payload> object;
              this->vs_ptr_->GetBulkStore()->Create(header.second, object_id, object);
              memcpy(object->pointer, header.first, header.second);
              for (int i = 0; i < header.second; i++) {
                std::cout << *(object->pointer + i);
              }
              std::cout << std::endl;
              LOG(INFO) << object_id;
              ret.header_offset = (uint64_t)object->pointer - (uint64_t) base_pointer;
              ret.header_size = object->data_size;
              LOG(INFO) << "header size: " << object->data_size;
              ret.type = OBJECT_TYPE::TENSOR;
              ret.ret = 0;
              // It should not be seen by client.
              // this->vs_ptr_->GetBulkStore()->Seal(object_id);
            }
          }
          VineyardSpinUnlock(&sync_val);
          return Status::OK();
        });
  } else if (param.type == OBJECT_TYPE::BLOB) {
    vs_ptr_->GetBulkStore()->GetUnsafe(ids, true, objects);
    for (auto iter = objects.begin(); iter != objects.end(); iter++) {
      pointer = (*iter)->pointer;
      file_size = (*iter)->data_size;
    }
    if (pointer) {
      ret.data_offset = (uint64_t) pointer - (uint64_t) base_pointer;
      ret.ret = 0;
      ret.data_size = file_size;
      ret.type = OBJECT_TYPE::BLOB;
    }
    VineyardSpinUnlock(&sync_val);
  } else {
    VineyardSpinUnlock(&sync_val);
  }
  // for sync test
  VineyardSpinLock(&sync_val);
  VineyardSpinUnlock(&sync_val);
  LOG(INFO) << "test";

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
  LOG(INFO) << __func__;
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
  case MSG_OPT::VINEYARD_READDIR:
    return HandleReadDir();
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
  int current_file_num;

  VineyardSpinLock(&obj_info_lock);
  current_file_num = header->total_file;
  header = reinterpret_cast<vineyard_object_info_header*>(obj_info_mem);
  entrys = reinterpret_cast<vineyard_entry*>(header + 1);
  current_file_num = header->total_file;
  PrintJsonElement(tree);

  if (type == OBJECT_TYPE::BLOB) {
    for (auto iter = tree.begin(); iter != tree.end(); iter++) {
      entrys[current_file_num + i].obj_id = ObjectIDFromString((*iter)["id"].get<std::string>());
      entrys[current_file_num + i].file_size = (*iter)["length"].get<uint64_t>();
      entrys[current_file_num + i].type = type;
      i++;
    }
  }

  if (type == OBJECT_TYPE::TENSOR) {
    for (auto iter = tree.begin(); iter != tree.end(); iter++) {
      LOG(INFO) << *iter;
      LOG(INFO) << ObjectIDFromString((*iter)["id"].get<std::string>());
      LOG(INFO) << (*iter)["nbytes"].get<uint64_t>();
      LOG(INFO) << ObjectIDFromString((*iter)["buffer_"]["id"].get<std::string>());
      
      entrys[header->total_file + i].obj_id = ObjectIDFromString((*iter)["id"].get<std::string>());
      entrys[header->total_file + i].file_size = (*iter)["nbytes"].get<uint64_t>();
      entrys[header->total_file + i].type = type;
      i++;
    }
  }
  header->total_file = current_file_num + i;
  LOG(INFO) << "total file " << header->total_file;
  VineyardSpinUnlock(&obj_info_lock);
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
    default:
      _fopt_ret = ns_ptr->HandleFops(&kmsg.msg.msg.request);
      memcpy(&umsg.ret._fopt_ret, &_fopt_ret, sizeof(fopt_ret));
      umsg.opt = kmsg.msg.msg.request.opt;
      break;
    }
  }

out:
  return;
}

}  // namespace vineyard
#endif  // __linux__
#endif  // BUILD_NETLINK_SERVER
