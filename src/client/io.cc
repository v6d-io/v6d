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

#include "client/io.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <iostream>

namespace vineyard {

static const int kNumConnectAttempts = 10;
static const int64_t kConnectTimeoutMs = 1000;

Status connect_ipc_socket(const std::string& pathname, int& socket_fd) {
  struct sockaddr_un socket_address;

  // vineyardd may use abstract socket address on Linux
  bool use_abstract_socket_address = false;
  std::string socket_pathname = pathname;
#ifdef __linux__
  if (pathname.size() > 0 && pathname[0] == '@') {
    use_abstract_socket_address = true;
  } else if (access(pathname.c_str(), F_OK | W_OK) != 0) {
    use_abstract_socket_address = true;
  }
  if (socket_pathname.size() > 0 && use_abstract_socket_address) {
    socket_pathname[0] = '@';
  }
#else
  if (access(socket_pathname.c_str(), F_OK | W_OK) != 0) {
    return Status::IOError("Cannot connect to " + socket_pathname + ": " +
                           strerror(errno));
  }
#endif

  socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (socket_fd < 0) {
    return Status::IOError("socket() failed for pathname " + socket_pathname +
                           ": " + strerror(errno));
  }

  memset(&socket_address, 0x00, sizeof(socket_address));
  socket_address.sun_family = AF_UNIX;
  if (socket_pathname.size() + 1 > sizeof(socket_address.sun_path)) {
    close(socket_fd);
    return Status::IOError("Socket pathname is too long: " + socket_pathname);
  }

  strncpy(socket_address.sun_path, socket_pathname.c_str(),
          socket_pathname.size() + 1);

  size_t socket_address_size = sizeof(socket_address);
  if (use_abstract_socket_address) {
    socket_address.sun_path[0] = '\0';
    // see also: https://stackoverflow.com/a/65435074
    socket_address_size =
        offsetof(struct sockaddr_un, sun_path) + socket_pathname.size();
  }

  if (connect(socket_fd, reinterpret_cast<struct sockaddr*>(&socket_address),
              socket_address_size) != 0) {
    close(socket_fd);
    return Status::IOError("connect() failed for pathname " + socket_pathname +
                           ": " + strerror(errno));
  }

  return Status::OK();
}

Status connect_rpc_socket(const std::string& host, uint32_t port,
                          int& socket_fd) {
  std::string port_string = std::to_string(port);

  struct addrinfo hints = {}, *addrs;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;

  if (getaddrinfo(host.c_str(), port_string.c_str(), &hints, &addrs) != 0) {
    return Status::IOError("getaddrinfo() failed for endpoint " + host + ":" +
                           std::to_string(port));
  }

  socket_fd = -1;
  for (struct addrinfo* addr = addrs; addr != nullptr; addr = addr->ai_next) {
    socket_fd = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (socket_fd == -1) {
      continue;
    }
    if (connect(socket_fd, addr->ai_addr, addr->ai_addrlen) != 0) {
      continue;
    }
    break;
  }
  freeaddrinfo(addrs);
  if (socket_fd == -1) {
    return Status::IOError("socket/connect failed for endpoint " + host + ":" +
                           std::to_string(port));
  }

  // avoid SIGPIPE in any cases as it is hard to catch, see also `send_bytes()`.
#if defined(__APPLE__)
  int option_value = 1;
  setsockopt(socket_fd, SOL_SOCKET, SO_NOSIGPIPE, &option_value,
             sizeof(option_value));
#endif

  return Status::OK();
}

Status connect_ipc_socket_retry(const std::string& pathname, int& socket_fd) {
  int num_retries = kNumConnectAttempts;
  int64_t timeout = kConnectTimeoutMs;

  auto status = connect_ipc_socket(pathname, socket_fd);

  while (!status.ok() && num_retries > 0) {
    std::clog << "[info] Connection to IPC socket failed for pathname "
              << pathname << " with ret = " << status << ", retrying "
              << num_retries << " more times." << std::endl;
    usleep(static_cast<int>(timeout * 1000));
    status = connect_ipc_socket(pathname, socket_fd);
    --num_retries;
  }
  if (!status.ok()) {
    status = Status::ConnectionFailed();
  }
  return status;
}

Status connect_rpc_socket_retry(const std::string& host, const uint32_t port,
                                int& socket_fd) {
  int num_retries = kNumConnectAttempts;
  int64_t timeout = kConnectTimeoutMs;

  auto status = connect_rpc_socket(host, port, socket_fd);

  while (!status.ok() && num_retries > 0) {
    std::clog << "[info] Connection to RPC socket failed for endpoint " << host
              << ":" << port << " with ret = " << status << ", retrying "
              << num_retries << " more times." << std::endl;
    usleep(static_cast<int>(timeout * 1000));
    status = connect_rpc_socket(host, port, socket_fd);
    --num_retries;
  }
  if (!status.ok()) {
    status = Status::ConnectionFailed();
  }
  return status;
}

Status send_bytes(int fd, const void* data, size_t length) {
  ssize_t nbytes = 0;
  size_t bytes_left = length;
  size_t offset = 0;
  const char* ptr = static_cast<const char*>(data);
  while (bytes_left > 0) {
    // NB: (in `Release()` operation) avoid SIGPIPE in any cases as it is hard
    // to catch and diagnose (the server may has already down).
    //
    // The `MSG_NOSIGNAL` is not supported on Mac, instead, we have set the flag
    // `SO_NOSIGPIPE` on the socket once established.
#if defined(__APPLE__)
    nbytes = write(fd, ptr + offset, bytes_left);
#else
    nbytes = send(fd, ptr + offset, bytes_left, MSG_NOSIGNAL);
#endif
    if (nbytes < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
        continue;
      }
      return Status::IOError("Send message failed: " +
                             std::string(strerror(errno)));
    } else if (nbytes == 0) {
      return Status::IOError("Send message failed: encountered unexpected EOF");
    }
    bytes_left -= nbytes;
    offset += nbytes;
  }
  return Status::OK();
}

Status send_message(int fd, const std::string& msg) {
  size_t length = msg.length();
  RETURN_ON_ERROR(send_bytes(fd, &length, sizeof(size_t)));
  RETURN_ON_ERROR(send_bytes(fd, msg.data(), length));
  return Status::OK();
}

Status recv_bytes(int fd, void* data, size_t length) {
  ssize_t nbytes = 0;
  size_t bytes_left = length;
  size_t offset = 0;
  char* ptr = static_cast<char*>(data);
  while (bytes_left > 0) {
    nbytes = read(fd, ptr + offset, bytes_left);
    if (nbytes < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
        continue;
      }
      return Status::IOError("Receive message failed: " +
                             std::string(strerror(errno)));
    } else if (nbytes == 0) {
      return Status::IOError(
          "Receive message failed: encountered unexpected EOF");
    }
    bytes_left -= nbytes;
    offset += nbytes;
  }
  return Status::OK();
}

Status recv_message(int fd, std::string& msg) {
  size_t length;
  RETURN_ON_ERROR(recv_bytes(fd, &length, sizeof(size_t)));
  msg.resize(length + 1);
  msg[length] = '\0';
  RETURN_ON_ERROR(recv_bytes(fd, &msg[0], length));
  return Status::OK();
}

Status check_fd(int fd) {
  int r = fcntl(fd, F_GETFL);
  if (r == -1) {
    return Status::Invalid("fd error.");
  } else if (r & O_RDONLY) {
    return Status::Invalid("fd is read-only.");
  } else if (r & O_WRONLY) {
    return Status::Invalid("fd is write-only.");
  }
  return Status::OK();
}
}  // namespace vineyard
