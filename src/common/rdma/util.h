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

#ifndef SRC_COMMON_RDMA_UTIL_H_
#define SRC_COMMON_RDMA_UTIL_H_

#include <unistd.h>
#include <string>

#include "common/util/status.h"
#if defined(__linux__)
#include "libfabric/include/rdma/fabric.h"
#endif  // defined(__linux__)

namespace vineyard {

#define CHECK_ERROR(condition, message) \
  if (!(condition)) {                   \
    return Status::Invalid(message);    \
  }

#if defined(__linux__)
#define POST(post_fn, op_str, ...)                                 \
  do {                                                             \
    int ret;                                                       \
    while (1) {                                                    \
      ret = post_fn(__VA_ARGS__);                                  \
      if (!ret) {                                                  \
        return Status::OK();                                       \
      }                                                            \
                                                                   \
      if (ret != -FI_EAGAIN) {                                     \
        std::cout << op_str << " " << ret << std::endl;            \
        std::string msg = "Failed to post " + std::string(op_str); \
        return Status::Invalid(msg);                               \
      }                                                            \
    }                                                              \
  } while (0)
#endif  // defined(__linux__)

enum VINEYARD_MSG_OPT {
  VINEYARD_MSG_EMPTY = 1,
  VINEYARD_MSG_CONNECT,
  VINEYARD_MSG_EXCHANGE_KEY,
  VINEYARD_MSG_REQUEST_MEM,
  VINEYARD_MSG_RELEASE_MEM,
  VINEYARD_MSG_CLOSE,
};

struct VineyardMsg {
  union {
    struct {
      uint64_t remote_address;
      uint64_t len;
      uint64_t key;
      void* mr_desc;
    } remoteMemInfo;
    struct {
      bool isReady;
    } ConnectState;
  };
  int type;
};

struct VineyardEventEntry {
  uint32_t event_id;
  void* fi;
  void* fid;
};

struct VineyardRecvContext {
  uint64_t rdma_conn_id;
  struct {
    void* msg_buffer;
  } attr;
};

struct VineyardSendContext {
  struct {
    void* msg_buffer;
  } attr;
};

struct RegisterMemInfo {
  uint64_t address;
  size_t size;
  uint64_t rkey;
  void* mr_desc;
  // TODO:
  // Work around for rpc client. Remove this field in the future
  void* mr;
};

enum RDMA_STATE {
  INIT,
  READY,
  STOPED,
};

struct RDMARemoteNodeInfo {
  void* fi;
  void* fabric;
  void* domain;
  int refcnt = 0;
};

#if defined(__linux__)
#define VINEYARD_FIVERSION FI_VERSION(1, 21)
#define VINEYARD_CONNREQ FI_CONNREQ
#define VINEYARD_CONNECTED FI_CONNECTED
#define VINEYARD_TX_CQ_DATA FI_SEND
#else
#define VINEYARD_CONNREQ 1
#define VINEYARD_CONNECTED 2
#define VINEYARD_TX_CQ_DATA (1ULL << 11)
#endif  // defined(__linux__)

}  // namespace vineyard

#endif  // SRC_COMMON_RDMA_UTIL_H_
