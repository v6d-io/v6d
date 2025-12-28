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

#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/dataframe.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "client/rpc_client.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

constexpr uint64_t total_mem = 1024UL * 1024 * 1024;

void PrepareData(std::vector<std::vector<std::shared_ptr<RemoteBlobWriter>>>&
                     remote_blob_writers_list,
                 size_t size, int parallel) {
  uint64_t iterator = total_mem / size;

  for (int i = 0; i < parallel; i++) {
    std::vector<std::shared_ptr<RemoteBlobWriter>> remote_blob_writers;
    for (size_t j = 0; j < iterator; j++) {
      auto remote_blob_writer = std::make_shared<RemoteBlobWriter>(size);
      uint8_t* data = reinterpret_cast<uint8_t*>(remote_blob_writer->data());
      for (size_t k = 0; k < size; k++) {
        data[k] = k % 256;
      }
      remote_blob_writers.push_back(remote_blob_writer);
    }
    remote_blob_writers_list.push_back(remote_blob_writers);
  }
}

void TestCreateBlob(
    std::shared_ptr<RPCClient>& client,
    std::vector<std::shared_ptr<RemoteBlobWriter>>& remote_blob_writers,
    std::vector<ObjectID>& ids, size_t size) {
  uint64_t iterator = total_mem / size;
  std::vector<ObjectMeta> metas;
  metas.reserve(iterator);
  LOG(INFO) << "Creating remote blobs...";

  auto start = std::chrono::high_resolution_clock::now();

  VINEYARD_CHECK_OK(client->CreateRemoteBlobs(remote_blob_writers, metas));

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG(INFO) << "Iterator: " << iterator << ", Size: " << size
            << ", Time: " << duration.count() << "us"
            << " average time:"
            << static_cast<double>(duration.count()) / iterator << "us";
  LOG(INFO) << "Speed:"
            << static_cast<double>(iterator * size) / 1024 / 1024 /
                   (static_cast<double>(duration.count()) / 1000 / 1000)
            << "MB/s\n";

  for (size_t i = 0; i < iterator; i++) {
    ids.push_back(metas[i].GetId());
  }
}

void TestGetBlob(std::shared_ptr<RPCClient>& client, std::vector<ObjectID>& ids,
                 size_t size,
                 std::vector<std::shared_ptr<RemoteBlob>>& local_buffers) {
  uint64_t iterator = total_mem / size;

  local_buffers.reserve(iterator);

  auto start = std::chrono::high_resolution_clock::now();
  VINEYARD_CHECK_OK(client->GetRemoteBlobs(ids, local_buffers));
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  LOG(INFO) << "Iterator: " << iterator << ", Size: " << size
            << ", Time: " << duration.count() << "us"
            << " average time:"
            << static_cast<double>(duration.count()) / iterator << "us";
  LOG(INFO) << "Speed:"
            << static_cast<double>(iterator * size) / 1024 / 1024 /
                   (static_cast<double>(duration.count()) / 1000 / 1000)
            << "MB/s\n";
}

void TestGetBlobWithAllocatedBuffer(
    std::shared_ptr<RPCClient>& client, std::vector<ObjectID>& ids, size_t size,
    std::vector<std::shared_ptr<MutableBuffer>>& local_buffers) {
  uint64_t iterator = total_mem / size;

  local_buffers.reserve(iterator);

  auto start = std::chrono::high_resolution_clock::now();
  VINEYARD_CHECK_OK(client->GetRemoteBlobs(ids, false, local_buffers));
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  LOG(INFO) << "Iterator: " << iterator << ", Size: " << size
            << ", Time: " << duration.count() << "us"
            << " average time:"
            << static_cast<double>(duration.count()) / iterator << "us";
  LOG(INFO) << "Speed:"
            << static_cast<double>(iterator * size) / 1024 / 1024 /
                   (static_cast<double>(duration.count()) / 1000 / 1000)
            << "MB/s\n";
}

void CheckBlobValue(std::vector<std::shared_ptr<RemoteBlob>>& local_buffers) {
  LOG(INFO) << "CheckBlobValue";
  for (size_t i = 0; i < local_buffers.size(); i++) {
    const uint8_t* data =
        reinterpret_cast<const uint8_t*>(local_buffers[i]->data());
    for (size_t j = 0; j < local_buffers[i]->size(); j++) {
      CHECK_EQ(data[j], j % 256);
    }
  }
  LOG(INFO) << "CheckBlobValue done";
}

void CheckBlobValue(
    std::vector<std::shared_ptr<MutableBuffer>>& local_buffers) {
  LOG(INFO) << "CheckBlobValue";
  for (size_t i = 0; i < local_buffers.size(); i++) {
    const uint8_t* data =
        reinterpret_cast<const uint8_t*>(local_buffers[i]->data());
    for (int64_t j = 0; j < local_buffers[i]->size(); j++) {
      CHECK_EQ(data[j], j % 256);
    }
  }
  LOG(INFO) << "CheckBlobValue done";
}

// Test 512K~512M blob
int main(int argc, const char** argv) {
  if (argc < 8) {
    LOG(ERROR) << "usage: " << argv[0] << " <ipc_socket>"
               << " <rpc_endpoint>"
               << " <rdma_endpoint>"
               << " <rdma_src_endpoint>"
               << " <min_size>"
               << " <max_size>"
               << " <parallel>";
    return -1;
  }
  std::string ipc_socket = std::string(argv[1]);
  std::string rpc_endpoint = std::string(argv[2]);
  std::string rdma_endpoint = std::string(argv[3]);
  std::string rdma_src_endpoint = std::string(argv[4]);
  std::cout << "rdma_src_endpoint: " << rdma_src_endpoint << std::endl;
  int parallel = std::stoi(argv[7]);
  std::vector<std::shared_ptr<RPCClient>> clients;
  for (int i = 0; i < parallel; i++) {
    clients.push_back(std::make_shared<RPCClient>());
    VINEYARD_CHECK_OK(clients[i]->Connect(
        rpc_endpoint, "", "", rdma_endpoint,
        rdma_src_endpoint + ":" + std::to_string(i + 5111)));
  }

  uint64_t min_size = 1024 * 1024 * 2;  // 512K
  uint64_t max_size = 1024 * 1024 * 2;  // 64M
  min_size = std::stoull(argv[5]) * 1024 * 1024;
  max_size = std::stoull(argv[6]) * 1024 * 1024;
  if (min_size == 0) {
    min_size = 1024 * 512;
  }
  if (max_size == 0) {
    max_size = 1024 * 512;
  }
  std::vector<std::vector<std::vector<ObjectID>>> blob_ids_lists;
  std::vector<size_t> sizes;

  LOG(INFO) << "Test Create Blob(RDMA write / TCP)";
  LOG(INFO) << "----------------------------";
  for (size_t size = min_size; size <= max_size; size *= 2) {
    std::vector<std::vector<ObjectID>> ids_list;
    ids_list.resize(parallel);
    std::vector<std::thread> threads;
    std::vector<std::vector<std::shared_ptr<RemoteBlobWriter>>>
        remote_blob_writers_list;
    PrepareData(remote_blob_writers_list, size, parallel);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < parallel; i++) {
      threads.push_back(std::thread(TestCreateBlob, std::ref(clients[i]),
                                    std::ref(remote_blob_writers_list[i]),
                                    std::ref(ids_list[i]), size));
    }
    for (int i = 0; i < parallel; i++) {
      threads[i].join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    uint64_t iterator = total_mem / size;
    LOG(INFO) << "Total time:" << duration.count() << "ms"
              << " average speed:"
              << static_cast<double>(iterator * size * parallel) / 1024 / 1024 /
                     (static_cast<double>(duration.count()) / 1000)
              << "MB/s\n";

    blob_ids_lists.push_back(ids_list);
    sizes.push_back(size);
  }

  std::vector<std::vector<std::vector<std::shared_ptr<RemoteBlob>>>>
      local_buffers_lists;
  LOG(INFO) << "Test Get Blob(RDMA read / TCP)";
  LOG(INFO) << "----------------------------";
  int index = 0;
  for (auto& blob_ids_list : blob_ids_lists) {
    std::vector<std::vector<std::shared_ptr<RemoteBlob>>> local_buffers_list;
    std::vector<std::thread> threads;
    local_buffers_list.resize(parallel);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < parallel; i++) {
      threads.push_back(std::thread(TestGetBlob, std::ref(clients[i]),
                                    std::ref(blob_ids_list[i]), sizes[index],
                                    std::ref(local_buffers_list[i])));
    }
    for (int i = 0; i < parallel; i++) {
      threads[i].join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    int iterator = total_mem / sizes[index];
    LOG(INFO) << "Total time:" << duration.count() << "ms"
              << " average speed:"
              << static_cast<double>(iterator * sizes[index] * parallel) /
                     1024 / 1024 /
                     (static_cast<double>(duration.count()) / 1000)
              << "MB/s\n";
    local_buffers_lists.push_back(local_buffers_list);
    index++;
  }

  for (auto& local_buffers_list : local_buffers_lists) {
    for (auto& local_buffers : local_buffers_list) {
      CheckBlobValue(local_buffers);
    }
  }

  LOG(INFO) << "Test Get Blob With Allocated Buffer(RDMA read / TCP)";
  LOG(INFO) << "----------------------------";
  std::vector<std::vector<std::vector<uint8_t*>>> buffers_lists;
  index = 0;
  std::vector<std::vector<std::vector<std::shared_ptr<MutableBuffer>>>>
      local_buffers_lists_3;
  for (auto& blob_ids_list : blob_ids_lists) {
    std::vector<std::vector<std::shared_ptr<MutableBuffer>>> local_buffers_list;
    std::vector<std::thread> threads;
    local_buffers_list.resize(parallel);
    std::vector<std::vector<uint8_t*>> buffers_list;
    for (int i = 0; i < parallel; i++) {
      std::vector<uint8_t*> buffers;
      local_buffers_list[i].resize(total_mem / sizes[index]);
      for (size_t j = 0; j < total_mem / sizes[index]; j++) {
        uint8_t* tmp_buffer = new uint8_t[sizes[index]];
        memset(tmp_buffer, 0, sizes[index]);
        local_buffers_list[i][j] =
            std::make_shared<MutableBuffer>(tmp_buffer, sizes[index]);
        buffers.push_back(tmp_buffer);
      }
      buffers_list.push_back(buffers);
    }
    buffers_lists.push_back(buffers_list);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < parallel; i++) {
      threads.push_back(std::thread(TestGetBlobWithAllocatedBuffer,
                                    std::ref(clients[i]),
                                    std::ref(blob_ids_list[i]), sizes[index],
                                    std::ref(local_buffers_list[i])));
    }
    for (int i = 0; i < parallel; i++) {
      threads[i].join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    int iterator = total_mem / sizes[0];
    LOG(INFO) << "Total time:" << duration.count() << "ms"
              << " average speed:"
              << static_cast<double>(iterator * sizes[index] * parallel) /
                     1024 / 1024 /
                     (static_cast<double>(duration.count()) / 1000)
              << "MB/s\n";
    local_buffers_lists_3.push_back(local_buffers_list);
  }

  for (auto& local_buffers_list : local_buffers_lists_3) {
    for (auto& local_buffers : local_buffers_list) {
      CheckBlobValue(local_buffers);
    }
  }

  LOG(INFO) << "Clean all object";
  for (size_t i = 0; i < blob_ids_lists.size(); i++) {
    for (size_t j = 0; j < blob_ids_lists[i].size(); j++) {
      VINEYARD_CHECK_OK(clients[j]->DelData(blob_ids_lists[i][j]));
    }
  }

  for (size_t i = 0; i < buffers_lists.size(); i++) {
    for (size_t j = 0; j < buffers_lists[i].size(); j++) {
      for (size_t k = 0; k < buffers_lists[i][j].size(); k++) {
        delete[] buffers_lists[i][j][k];
      }
    }
  }

  for (int i = 0; i < parallel; i++) {
    clients[i]->Disconnect();
  }

  LOG(INFO) << "Passed blob rdma performance test.";

  return 0;
}
