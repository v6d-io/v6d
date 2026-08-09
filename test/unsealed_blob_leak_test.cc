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

#include "client/client.h"
#include "client/ds/blob.h"
#include "common/util/logging.h"
#include "common/util/status.h"

using namespace vineyard;  // NOLINT(build/namespaces)

constexpr size_t blob_size = 4 * 1024 * 1024;

static size_t current_memory_usage(Client& client) {
  std::shared_ptr<struct InstanceStatus> status;
  VINEYARD_CHECK_OK(client.InstanceStatus(status));
  return status->memory_usage;
}

// Wait until the server observes the disconnection and finishes the cleanup,
// as the connection teardown happens asynchronously on the server side.
static bool wait_memory_usage_until(Client& client, const size_t expected) {
  for (int retry = 0; retry < 100; ++retry) {
    if (current_memory_usage(client) == expected) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return false;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./unsealed_blob_leak_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client observer;
  VINEYARD_CHECK_OK(observer.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  const size_t baseline = current_memory_usage(observer);

  {  // created-but-unsealed blobs are reclaimed on disconnection
    Client client;
    VINEYARD_CHECK_OK(client.Connect(ipc_socket));
    std::unique_ptr<BlobWriter> blob_writer;
    VINEYARD_CHECK_OK(client.CreateBlob(blob_size, blob_writer));
    CHECK_GE(current_memory_usage(observer), baseline + blob_size);
    client.Disconnect();
    CHECK(wait_memory_usage_until(observer, baseline));
  }

  {  // blobs dropped before disconnection don't trigger double-free
    Client client;
    VINEYARD_CHECK_OK(client.Connect(ipc_socket));
    std::unique_ptr<BlobWriter> blob_writer;
    VINEYARD_CHECK_OK(client.CreateBlob(blob_size, blob_writer));
    VINEYARD_CHECK_OK(blob_writer->Abort(client));
    client.Disconnect();
    CHECK(wait_memory_usage_until(observer, baseline));
  }

  ObjectID sealed_blob_id = InvalidObjectID();
  {  // sealed blobs survive the disconnection of their creator
    Client client;
    VINEYARD_CHECK_OK(client.Connect(ipc_socket));
    std::unique_ptr<BlobWriter> blob_writer;
    VINEYARD_CHECK_OK(client.CreateBlob(blob_size, blob_writer));
    std::shared_ptr<Object> blob_object;
    VINEYARD_CHECK_OK(blob_writer->Seal(client, blob_object));
    sealed_blob_id = blob_object->id();
    client.Disconnect();

    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::shared_ptr<Blob> blob;
    VINEYARD_CHECK_OK(observer.GetBlob(sealed_blob_id, blob));
    CHECK_GE(current_memory_usage(observer), baseline + blob_size);
  }

  {  // cleanup, and check the sealed blob can be deleted as usual
    VINEYARD_CHECK_OK(observer.DelData(sealed_blob_id));
  }

  LOG(INFO) << "Passed unsealed blob leak tests...";

  observer.Disconnect();

  return 0;
}
