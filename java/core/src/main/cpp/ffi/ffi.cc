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
#include "io_v6d_core_common_memory_ffi_Fling.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

static void init_msg(struct msghdr* msg, struct iovec* iov, char* buf,
              size_t buf_len) {
  iov->iov_base = buf;
  iov->iov_len = 1;

  msg->msg_iov = iov;
  msg->msg_iovlen = 1;
  msg->msg_control = buf;
  msg->msg_controllen = static_cast<socklen_t>(buf_len);
  msg->msg_name = NULL;
  msg->msg_namelen = 0;
}

static int recv_fd(int conn) {
  struct msghdr msg;
  struct iovec iov;
  char buf[CMSG_SPACE(sizeof(int))];
  init_msg(&msg, &iov, buf, sizeof(buf));

  while (true) {
    ssize_t r = recvmsg(conn, &msg, 0);
    if (r == -1) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
        continue;
      } else {
        fprintf(stderr, "[error] Error in recv_fd (errno = %d: %s)\n", errno, strerror(errno));
        return -1;
      }
    } else {
      break;
    }
  }

  int found_fd = -1;
  int oh_noes = 0;
  for (struct cmsghdr* header = CMSG_FIRSTHDR(&msg); header != NULL;
       header = CMSG_NXTHDR(&msg, header))
    if (header->cmsg_level == SOL_SOCKET && header->cmsg_type == SCM_RIGHTS) {
      ssize_t count =
          (header->cmsg_len -
           (CMSG_DATA(header) - reinterpret_cast<unsigned char*>(header))) /
          sizeof(int);
      for (int i = 0; i < count; ++i) {
        int fd = (reinterpret_cast<int*>(CMSG_DATA(header)))[i];
        if (found_fd == -1) {
          found_fd = fd;
        } else {
          close(fd);
          oh_noes = 1;
        }
      }
    }

  // The sender sent us more than one file descriptor. We've closed
  // them all to prevent fd leaks but notify the caller that we got
  // a bad message.
  if (oh_noes) {
    close(found_fd);
    errno = EBADMSG;
    fprintf(stderr, "[error] Error in recv_fd: more than one fd received in message\n");
    return -1;
  }

  return found_fd;
}

/*
 * Class:     io_v6d_core_common_memory_ffi_Fling
 * Method:    sendFD
 * Signature: (II)I
 */
JNIEXPORT jint JNICALL Java_io_v6d_core_common_memory_ffi_Fling_sendFD(
    JNIEnv*, jclass, jint conn, jint fd) {
  // return send_fd(conn, fd);
  return -1;
}

/*
 * Class:     io_v6d_core_common_memory_ffi_Fling
 * Method:    recvFD
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL
Java_io_v6d_core_common_memory_ffi_Fling_recvFD(JNIEnv*, jclass, jint conn) {
  return recv_fd(conn);
}

/*
 * Class:     io_v6d_core_common_memory_ffi_Fling
 * Method:    mapSharedMem
 * Signature: (IJZZ)J
 */
JNIEXPORT jlong JNICALL Java_io_v6d_core_common_memory_ffi_Fling_mapSharedMem(
    JNIEnv*, jclass, jint fd, jlong map_size, jboolean readonly,
    jboolean realign) {
  size_t length = map_size;
  if (realign == JNI_TRUE) {
    length -= sizeof(size_t);
  }
  void* pointer = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (pointer == MAP_FAILED) {
    fprintf(stderr, "mmap failed: errno = %d: %s\n", errno, strerror(errno));
  }
  return reinterpret_cast<jlong>(pointer);
}

#ifdef __cplusplus
}
#endif
