/** Copyright 2020 Alibaba Group Holding Limited.

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
#include "common/util/logging.h"

namespace vineyard {

static const int kNumConnectAttempts = 10;
static const int64_t kConnectTimeoutMs = 1000;

Status connect_ipc_socket(const std::string& pathname, int& socket_fd) {
  struct sockaddr_un socket_address;

  if (access(pathname.c_str(), F_OK | W_OK) != 0) {
    return Status::IOError("Cannot connect to " + pathname + ": " +
                           strerror(errno));
  }

  socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (socket_fd < 0) {
    return Status::IOError("socket() failed for pathname " + pathname);
  }

  memset(&socket_address, 0, sizeof(socket_address));
  socket_address.sun_family = AF_UNIX;
  if (pathname.size() + 1 > sizeof(socket_address.sun_path)) {
    close(socket_fd);
    return Status::IOError("Socket pathname is too long: " + pathname);
  }

  strncpy(socket_address.sun_path, pathname.c_str(), pathname.size() + 1);

  if (connect(socket_fd, reinterpret_cast<struct sockaddr*>(&socket_address),
              sizeof(socket_address)) != 0) {
    close(socket_fd);
    return Status::IOError("connect() failed for pathname " + pathname);
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

  return Status::OK();
}

Status connect_ipc_socket_retry(const std::string& pathname, int& socket_fd) {
  int num_retries = kNumConnectAttempts;
  int64_t timeout = kConnectTimeoutMs;

  auto status = connect_ipc_socket(pathname, socket_fd);

  while (!status.ok() && num_retries > 0) {
    VLOG(2) << "Connection to IPC socket failed for pathname " << pathname
            << " with ret = " << status << ", retrying " << num_retries
            << " more times.";
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
    VLOG(2) << "Connection to RPC socket failed for endpoint " << host << ":"
            << port << " with ret = " << status << ", retrying " << num_retries
            << " more times.";
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
    nbytes = write(fd, ptr + offset, bytes_left);
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

}  // namespace vineyard
