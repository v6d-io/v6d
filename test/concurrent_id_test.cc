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

#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common/util/logging.h"
#include "common/util/status.h"

#include "client/client.h"
#include "client/ds/blob.h"

using namespace vineyard;  // NOLINT(build/namespaces)

const int num_threads = 16;
const int ids_per_thread = 10000;

void testGenerateBlobID(std::string ipc_socket) {
  std::unordered_set<uint64_t> blob_ids;
  std::mutex mtx;

  auto generate_blob_id = [&]() {
    auto client = std::make_shared<Client>();
    VINEYARD_CHECK_OK(client->Connect(ipc_socket));
    for (int i = 0; i < ids_per_thread; ++i) {
      std::unique_ptr<BlobWriter> blob_writer;

      VINEYARD_CHECK_OK(client->CreateBlob(1, blob_writer));
      auto blob_id = blob_writer->id();
      std::lock_guard<std::mutex> lock(mtx);
      auto result = blob_ids.insert(blob_id);
      if (!result.second) {
        LOG(ERROR) << "Duplicated blob id: " << blob_id;
      }
      CHECK(result.second == true);
    }
    VINEYARD_CHECK_OK(client->Clear());
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(generate_blob_id);
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

void testGenerateObjectID(std::string ipc_socket) {
  std::unordered_set<uint64_t> object_ids;
  std::mutex mtx;

  auto generate_object_id = [&]() {
    auto client = std::make_shared<Client>();
    VINEYARD_CHECK_OK(client->Connect(ipc_socket));
    for (int i = 0; i < ids_per_thread; ++i) {
      std::unique_ptr<BlobWriter> blob_writer;
      VINEYARD_CHECK_OK(client->CreateBlob(1, blob_writer));
      std::shared_ptr<Object> object = blob_writer->Seal(*client.get());
      auto object_id = object->id();
      std::lock_guard<std::mutex> lock(mtx);
      auto result = object_ids.insert(object_id);
      if (!result.second) {
        LOG(ERROR) << "Duplicated object id: " << object_id;
      }
      CHECK(result.second == true);
    }
    VINEYARD_CHECK_OK(client->Clear());
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(generate_object_id);
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./concurrent_id_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  testGenerateBlobID(ipc_socket);
  testGenerateObjectID(ipc_socket);
  return 0;
}
