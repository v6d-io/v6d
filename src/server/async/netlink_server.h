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

struct fopt_param {
  // read/write/sync
  uint64_t          obj_id;
  uint64_t          offset;
  // open
  enum OBJECT_TYPE  type;
  uint64_t          length;
};

struct vineyard_request_msg {
  enum REQUEST_OPT  opt;
  struct fopt_param _fopt_param;
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
  int           has_msg;
  unsigned int  lock;
  int           head_point;
  int           tail_point;
  int           close;
};

struct vineyard_result_mem_header {
  int           has_msg;
  unsigned int  lock;
  int           head_point;
  int           tail_point;
};

struct vineyard_rw_lock {
  unsigned int r_lock;
  unsigned int w_lock;
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

static inline void VineyardSpinLock(volatile unsigned int *addr)
{
  while(!__sync_bool_compare_and_swap(addr, 0, 1));
}

static inline void VineyardSpinUnlock(volatile unsigned int *addr)
{
  *addr = 0;
}

static inline void VineyardWriteLock(volatile unsigned int *rlock, volatile unsigned int *wlock)
{
  VineyardSpinLock(wlock);
  while(*rlock);
}

static inline void VineyardWriteUnlock(volatile unsigned int *wlock)
{
  *wlock = 0;
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

  void SyncObjectEntryList();

 private:
  void InitNetLink();

  void InitialBulkField();

  void doAccept() override;

  int HandleSet(struct vineyard_kern_user_msg *msg);

  int HandleOpen();

  int HandleRead(fopt_param &param);

  int HandleWrite();

  int HandleCloseOrFsync();

  int HandleReadDir();

  int HandleFops();

  void FillFileEntryInfo(const json &tree, enum OBJECT_TYPE type);

  int ReadRequestMsg();

  int WriteResultMsg();

  bool VineyardGetRequestMsg(vineyard_msg_mem_header *header, vineyard_request_msg *msg);

  int VineyardSetResultMsg(vineyard_result_msg &rmsg);

  static void thread_routine(NetLinkServer *ns_ptr, int socket_fd, struct sockaddr_nl saddr, struct sockaddr_nl daddr, struct nlmsghdr *nlh);

  int socket_fd;
  struct sockaddr_nl saddr, daddr;
  struct nlmsghdr *nlh;
  std::thread *work;

  void *req_mem;
  void *result_mem;
  void *obj_info_mem;

  uint64_t base_object_id;
  void *base_pointer;
};/*  */

}

#endif
