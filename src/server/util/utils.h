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

#ifndef SRC_SERVER_UTIL_UTILS_H_
#define SRC_SERVER_UTIL_UTILS_H_

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

namespace vineyard {

#ifndef SECOND_TO_MILLISECOND
#define SECOND_TO_MILLISECOND(x) ((x) *1000)
#endif

static inline bool is_port_available(int port) {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    return false;
  }

  int opt = 1;
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  if (bind(sock, (struct sockaddr*) &addr, sizeof(addr)) == 0) {
    close(sock);
    return true;
  } else {
    close(sock);
    return false;
  }
}

}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_UTILS_H_
