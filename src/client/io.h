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

#ifndef SRC_CLIENT_IO_H_
#define SRC_CLIENT_IO_H_

#include <string>

#include "common/util/status.h"

namespace vineyard {

Status connect_ipc_socket(const std::string& pathname, int& socket_fd);

Status connect_rpc_socket(const std::string& host, const uint32_t port,
                          int& socket_fd);

Status connect_ipc_socket_retry(const std::string& pathname, int& socket_fd);

Status connect_rpc_socket_retry(const std::string& host, const uint32_t port,
                                int& socket_fd);

Status send_bytes(int fd, const void* data, size_t length);

Status send_message(int fd, const std::string& msg);

Status recv_bytes(int fd, void* data, size_t length);

Status recv_message(int fd, std::string& msg);

Status check_fd(int fd);

}  // namespace vineyard

#endif  // SRC_CLIENT_IO_H_
