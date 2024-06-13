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
// #define VINEYARD_WITH_RDMA
#ifdef VINEYARD_WITH_RDMA

#include <sys/mman.h>

#include <chrono>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>

#include "common/rdma/rdma_client.h"
#include "common/rdma/rdma_server.h"
#include "common/memory/memcpy.h"

int register_round = 1000;
uint64_t register_size = 1024 * 1024 * 1024;
#define BUFFER_NUM 2

using namespace vineyard; // NOLINT(build/namespaces)

void testRDMARegisterPerf() {
  std::shared_ptr<RDMAServer> server;
  VINEYARD_CHECK_OK(RDMAServer::Make(server, 2223));
  void *buffer = nullptr;
  fid_mr* mr;
  void *mr_desc;
  uint64_t rkey;

  buffer = mmap(NULL, register_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  if (buffer == MAP_FAILED) {
    LOG(ERROR) << "mmap failed.";
    return;
  }

  // warm up
  // for (int i = 0; i < 50; i++) {
  //   server->RegisterMemory(&mr, buffer, register_size, rkey, mr_desc);
  //   server->DeregisterMemory(mr);
  // }
  if(mlock(buffer, register_size)!=0) {
    LOG(ERROR) << "mlock failed.";
    return;
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < register_round; i++) {
    VINEYARD_CHECK_OK(server->RegisterMemory(&mr, buffer, register_size, rkey, mr_desc));
    VINEYARD_CHECK_OK(server->DeregisterMemory(mr));
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  LOG(INFO) << "RegisterMemory and DeregisterMemory cost: " << diff.count() << " ms for " << register_round << " rounds."
            << " Per round cost: " << diff.count() / register_round << " ms.";
  munlock(buffer, register_size);
  munmap(buffer, register_size);
}

void testRDMARegisterPerf_2() {
  std::shared_ptr<RDMAServer> server;
  VINEYARD_CHECK_OK(RDMAServer::Make(server, 2224));
  void *buffer[BUFFER_NUM];
  fid_mr* mr;
  void *mr_desc;
  uint64_t rkey;

  LOG(INFO) << "Map " << BUFFER_NUM << " buffers. size:" << (double)register_size / BUFFER_NUM / 1024 / 1024 / 1024 << " GB.";
  for (int i = 0; i < BUFFER_NUM; i++) {
    buffer[i] = mmap(NULL, register_size / BUFFER_NUM, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (buffer[i] == MAP_FAILED) {
      LOG(ERROR) << "mmap failed.";
      return;
    }
    if(mlock(buffer[i], register_size / BUFFER_NUM)!=0) {
      LOG(ERROR) << "mlock failed.";
      return;
    }
  }

  // warm up
  for (int i = 0; i < 50; i++) {
    server->RegisterMemory(&mr, buffer[(i % BUFFER_NUM)], register_size / BUFFER_NUM, rkey, mr_desc);
    server->DeregisterMemory(mr);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < register_round; i++) {
    VINEYARD_CHECK_OK(server->RegisterMemory(&mr, buffer[(i % BUFFER_NUM)], register_size / BUFFER_NUM, rkey, mr_desc));
    VINEYARD_CHECK_OK(server->DeregisterMemory(mr));
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  LOG(INFO) << "RegisterMemory and DeregisterMemory cost: " << diff.count() << " ms for " << register_round << " rounds."
            << " Per round cost: " << diff.count() / register_round << " ms.";
  for (int i = 0; i < BUFFER_NUM; i++) {
    mlock(buffer[i], register_size / BUFFER_NUM);
    munmap(buffer[i], register_size / BUFFER_NUM);
  }
}

void testMemcopy() {
  void *buffer = mmap(NULL, register_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  void *buffer2 = mmap(NULL, register_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  if (buffer == MAP_FAILED || buffer2 == MAP_FAILED) {
    LOG(ERROR) << "mmap failed.";
    return;
  }
  memset(buffer, 1000, register_size);
  // warm up
  for (int i = 0; i < 10; i++) {
    memcpy(buffer2, buffer, register_size);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < register_round; i++) {
    memcpy(buffer2, buffer, register_size);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  LOG(INFO) << "Memcpy cost: " << diff.count() << " ms for " << register_round << " rounds."
            << " Per round cost: " << diff.count() / register_round << " ms.";
  munmap(buffer, register_size);
  munmap(buffer2, register_size);
}

void testMemcopyMultiThread() {
  LOG(INFO) << "multi thread";
  void *buffer = mmap(NULL, register_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  void *buffer2 = mmap(NULL, register_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  if (buffer == MAP_FAILED || buffer2 == MAP_FAILED) {
    LOG(ERROR) << "mmap failed.";
    return;
  }
  memset(buffer, 1000, register_size);
  // warm up
  for (int i = 0; i < 10; i++) {
    memcpy(buffer2, buffer, register_size);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < register_round; i++) {
    memory::concurrent_memcpy(buffer2, buffer, register_size, 32);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  LOG(INFO) << "Memcpy cost: " << diff.count() << " ms for " << register_round << " rounds."
            << " Per round cost: " << diff.count() / register_round << " ms.";
  munmap(buffer, register_size);
  munmap(buffer2, register_size);
}

void testMadvice() {
  void *buffer = mmap(NULL, register_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  if (buffer == MAP_FAILED) {
    LOG(ERROR) << "mmap failed.";
    return;
  }
  LOG(INFO) << "Test madvice";
  sleep(2);
  madvise(buffer, register_size, MADV_DONTNEED);
  LOG(INFO) << "Test madvice done.";
  munmap(buffer, register_size);
}
#endif

int main(int argc, char** argv) {
#ifdef VINEYARD_WITH_RDMA
  if (argc > 1) {
    register_round = std::stoi(argv[1]);
  }
  if (argc > 2) {
    register_size = std::stoull(argv[2]);
  }
  LOG(INFO) << "Register round: " << register_round << ", register size: " << (double)register_size / 1024 / 1024 / 1024 << " GB.";
  // testRDMARegisterPerf();
  testRDMARegisterPerf_2();
  // testMemcopy();
  // testMemcopyMultiThread();
  // testMadvice();
#endif

  return 0;
}