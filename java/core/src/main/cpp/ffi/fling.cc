/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#include "vineyard/common/memory/fling.h"

/*
 * Class:     io_v6d_core_common_memory_ffi_Fling
 * Method:    sendFD
 * Signature: (II)I
 */
JNIEXPORT jint JNICALL Java_io_v6d_core_common_memory_ffi_Fling_sendFD
  (JNIEnv *, jclass, jint conn, jint fd) {
  return send_fd(conn, fd);
}

/*
 * Class:     io_v6d_core_common_memory_ffi_Fling
 * Method:    recvFD
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_io_v6d_core_common_memory_ffi_Fling_recvFD
  (JNIEnv *, jclass, jint conn) {
  return recv_fd(conn);
}
