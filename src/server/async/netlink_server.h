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

enum REQUEST_TYPE {
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
    enum REQUEST_TYPE type;
  } _fopt_param;
};

struct vineyard_kern_user_msg {
    enum USER_KERN_OPT  opt;
    uint64_t            request_mem;
    uint64_t            result_mem;
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

static inline bool msg_empty(int head_point, int tail_point)
{
  return head_point == tail_point;
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

 private:
  void doAccept() override;

  static void thread_routine(NetLinkServer *ns_ptr, int socket_fd, struct sockaddr_nl saddr, struct sockaddr_nl daddr, struct nlmsghdr *nlh);

  int HandleSet(struct vineyard_kern_user_msg *msg);

  int HandleOpen();

  int HandleRead();

  int HandleWrite();

  int HandleCloseAndFsync();

  int socket_fd;
  struct sockaddr_nl saddr, daddr;
  struct nlmsghdr *nlh;
  std::thread *work;
  void *req_mem;
  void *result_mem;
};/*  */

}

#endif
