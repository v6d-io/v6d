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

#include <sys/mman.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "client/client.h"
#include "common/util/env.h"
#include "common/util/logging.h"

#if defined(WITH_MIMALLOC)
#include "malloc/mimalloc_allocator.h"
#endif

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./allocator_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

#if defined(WITH_MIMALLOC)
  VineyardMimallocAllocator<void>* allocator =
      VineyardMimallocAllocator<void>::Create(client);

  void* p1 = allocator->allocate(1025);
  allocator->deallocate(p1);

  void* p2 = allocator->allocate(1026);
  allocator->Freeze(p2);

  VINEYARD_CHECK_OK(allocator->Release());
#endif

  LOG(INFO) << "Passed allocator tests...";

  client.Disconnect();

  return 0;
}
