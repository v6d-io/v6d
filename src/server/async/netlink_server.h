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
#ifndef SRC_SERVER_ASYNC_NL_SERVER_H_
#define SRC_SERVER_ASYNC_NL_SERVER_H_

#include <memory>
#include <string>
#include <linux/netlink.h>
#include <sys/socket.h>

#include "boost/asio.hpp"

#include "common/util/protocols.h"
#include "server/async/socket_server.h"
#include "server/memory/memory.h"

namespace vineyard {

class VineyardServer;

#define NETLINK_VINEYARD  22
#define NETLINK_PORT      100

enum OBJECT_TYPE {
  BLOB = 1,
};

enum REQUEST_OPT {
  OPEN,
  READ,
  WRITE,
  CLOSE,
  FSYNC,
};

enum USER_KERN_OPT {
  SET,
  INIT,
  WAIT,
  FOPT,
  EXIT,
};

struct vineyard_result_msg {
  enum USER_KERN_OPT opt;
  uint64_t        obj_id;
  uint64_t        offset;
  uint64_t        size;
  int             ret;
};

struct vineyard_request_msg {
  enum REQUEST_OPT  opt;
  struct fopt_param {
    // read/write/sync
    uint64_t          obj_id;
    uint64_t          offset;
    // open
    enum OBJECT_TYPE type;
  } _fopt_param;
};

struct vineyard_kern_user_msg {
    enum USER_KERN_OPT  opt;
    uint64_t            request_mem;
    uint64_t            result_mem;
    uint64_t            obj_info_mem;
};

struct kmsg {
  struct nlmsghdr hdr;
  struct vineyard_kern_user_msg msg;
};

struct vineyard_msg_mem_header {
  int has_msg;
  int lock;
  int head_point;
  int tail_point;
  int close;
};

struct vineyard_result_mem_header {
  int has_msg;
  int lock;
  int head_point;
  int tail_point;
};

struct vineyard_rw_lock {
    int r_lock;
    int w_lock;
};

struct vineyard_object_info_header {
    struct vineyard_rw_lock rw_lock;
    int total_file;
};

struct vineyard_entry {
	uint64_t			obj_id; // as name
	uint64_t			file_size;
	enum OBJECT_TYPE	type;
	unsigned long 		inode_id;
};

static inline bool MsgEmpty(int head_point, int tail_point)
{
  return head_point == tail_point;
}

static inline void vineyard_spin_lock(volatile int *addr)
{
  while(!__sync_bool_compare_and_swap(addr, 0, 1));
}

static inline void vineyard_spin_unlock(volatile int *addr)
{
  *addr = 0;
}

static inline void vineyard_write_lock(volatile int *rlock, volatile int *wlock)
{
  vineyard_spin_lock(wlock);
  while(*rlock);
}

static inline void vineyard_write_unlock(volatile int *wlock)
{
  *wlock = 0;
}

static vineyard_request_msg *vineyard_get_request_msg(vineyard_msg_mem_header *header)
{
  struct vineyard_request_msg *entrys;
  struct vineyard_request_msg *entry = NULL;

  entrys = (struct vineyard_request_msg *)(header + 1);
  vineyard_spin_lock(&header->lock);
  if (!MsgEmpty(header->head_point, header->tail_point)) {
    entry = &(entrys[header->tail_point]);
    header->tail_point++;
    // TODO: ring buffer reset pointer.
  }
  vineyard_spin_unlock(&header->lock);

  return entry;
}

class NetLinkServer : public SocketServer,
                      public std::enable_shared_from_this<NetLinkServer>{
 public:
  explicit NetLinkServer(std::shared_ptr<VineyardServer> vs_ptr);

  ~NetLinkServer() override;

  void Start() override;

  void Close() override;

  std::string Socket() {
    return std::string("");
  }

  void RefreshObjectList();

 private:
  void doAccept() override;

  static void thread_routine(NetLinkServer *ns_ptr, int socket_fd, struct sockaddr_nl saddr, struct sockaddr_nl daddr, struct nlmsghdr *nlh);

  int HandleSet(struct vineyard_kern_user_msg *msg);

  int HandleOpen();

  int HandleRead();

  int HandleWrite();

  int HandleCloseOrFsync();

  int HandleReadDir();

  int HandleFops();

  void FillFileMsg(const json &tree, enum OBJECT_TYPE type);

  int ReadRequestMsg();

  int WriteResultMsg();

  int socket_fd;
  struct sockaddr_nl saddr, daddr;
  struct nlmsghdr *nlh;
  std::thread *work;
  void *req_mem;
  void *result_mem;
  void *obj_info_mem;
};/*  */

}

#endif
