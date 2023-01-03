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

#include "vineyard/common/memory/fling.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     io_v6d_core_common_memory_ffi_Fling
 * Method:    sendFD
 * Signature: (II)I
 */
JNIEXPORT jint JNICALL Java_io_v6d_core_common_memory_ffi_Fling_sendFD(
    JNIEnv*, jclass, jint conn, jint fd) {
  return send_fd(conn, fd);
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
