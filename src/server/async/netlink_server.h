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
#ifndef SRC_SERVER_ASYNC_NETLINK_SERVER_H_
#define SRC_SERVER_ASYNC_NETLINK_SERVER_H_

#include <memory>
#include <string>

#include "boost/asio.hpp"

#include "common/util/protocols.h"
#include "server/async/socket_server.h"
#include "server/memory/memory.h"

#ifdef __linux__
#include <linux/netlink.h>
#include <sys/socket.h>

namespace vineyard {

class VineyardServer;

// It is a user value.
#define NETLINK_VINEYARD 23
#define NETLINK_PORT 101

enum OBJECT_TYPE {
  BLOB = 1,
};

enum MSG_OPT {
  VINEYARD_WAIT,
  VINEYARD_MOUNT,
  VINEYARD_SET_BULK_ADDR,
  VINEYARD_EXIT,
  VINEYARD_OPEN,
  VINEYARD_READ,
  VINEYARD_WRITE,
  VINEYARD_CLOSE,
  VINEYARD_FSYNC,
};

struct fopt_ret {
  uint64_t obj_id;
  uint64_t offset;
  uint64_t size;
  int ret;
};

struct set_ret {
  uint64_t bulk_addr;
  uint64_t bulk_size;
  int ret;
};

struct vineyard_result_msg {
  enum MSG_OPT opt;
  int has_msg;
  union {
    struct fopt_ret _fopt_ret;
    struct set_ret _set_ret;
  } ret;
};

struct fopt_param {
  // read/write/sync
  uint64_t obj_id;
  uint64_t offset;
  // open
  enum OBJECT_TYPE type;
  uint64_t length;
};

struct set_param {
  uint64_t obj_info_mem;
};

struct vineyard_request_msg {
  enum MSG_OPT opt;
  int has_msg;
  union {
    struct fopt_param _fopt_param;
    struct set_param _set_param;
  } param;
};

struct vineyard_rw_lock {
  unsigned int r_lock;
  unsigned int w_lock;
};

struct vineyard_object_info_header {
  struct vineyard_rw_lock rw_lock;
  int total_file;
};

struct vineyard_msg {
  union {
    struct vineyard_request_msg request;
    struct vineyard_result_msg result;
  } msg;
};

struct kmsg {
  struct nlmsghdr hdr;
  struct vineyard_msg msg;
};

struct vineyard_entry {
  uint64_t obj_id;  // as name
  uint64_t file_size;
  enum OBJECT_TYPE type;
  uint64_t inode_id;
};

static inline void VineyardSpinLock(volatile unsigned int* addr) {
  while (!__sync_bool_compare_and_swap(addr, 0, 1)) {}
}

static inline void VineyardSpinUnlock(volatile unsigned int* addr) {
  *addr = 0;
}

static inline void VineyardWriteLock(volatile unsigned int* rlock,
                                     volatile unsigned int* wlock) {
  VineyardSpinLock(wlock);
  while (*rlock) {}
}

static inline void VineyardWriteUnlock(volatile unsigned int* wlock) {
  *wlock = 0;
}

class NetLinkServer : public SocketServer,
                      public std::enable_shared_from_this<NetLinkServer> {
 public:
  explicit NetLinkServer(std::shared_ptr<VineyardServer> vs_ptr);

  ~NetLinkServer() override;

  void Start() override;

  void Close() override;

  std::string Socket() { return std::string(""); }

  void SyncObjectEntryList();

 private:
  void InitNetLink();

  uint64_t GetServerBulkField();

  uint64_t GetServerBulkSize();

  void InitialBulkField();

  void doAccept() override;

  int HandleSet(struct vineyard_request_msg* msg);

  fopt_ret HandleOpen(fopt_param& param);

  fopt_ret HandleRead(fopt_param& param);

  fopt_ret HandleWrite();

  fopt_ret HandleCloseOrFsync();

  fopt_ret HandleReadDir();

  fopt_ret HandleFops(vineyard_request_msg* msg);

  void FillFileEntryInfo(const json& tree, enum OBJECT_TYPE type);

  int ReadRequestMsg();

  int WriteResultMsg();

  static void thread_routine(NetLinkServer* ns_ptr, int socket_fd,
                             struct sockaddr_nl saddr, struct sockaddr_nl daddr,
                             struct nlmsghdr* nlh);

  int socket_fd;
  struct sockaddr_nl saddr, daddr;
  struct nlmsghdr* nlh;
  std::thread* work;

  void* obj_info_mem;

  uint64_t base_object_id;
  void* base_pointer;
}; /*  */

}  // namespace vineyard
#else
namespace vineyard {
class NetLinkServer : public SocketServer,
                      public std::enable_shared_from_this<NetLinkServer> {
 public:
  void SyncObjectEntryList() {}
};
}  // namespace vineyard

#endif
#endif  // SRC_SERVER_ASYNC_NETLINK_SERVER_H_
