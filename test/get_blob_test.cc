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

#include <memory>
#include <string>
#include <thread>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/array.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./get_blob_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client1;
  Client client2;
  VINEYARD_CHECK_OK(client1.Connect(ipc_socket));
  VINEYARD_CHECK_OK(client2.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  std::unique_ptr<BlobWriter> blob_writer;
  VINEYARD_CHECK_OK(client1.CreateBlob(1024, blob_writer));
  VINEYARD_ASSERT(blob_writer != nullptr);

  std::shared_ptr<Blob> blob;

  // safe get
  CHECK(client2.GetBlob(blob_writer->id(), blob).IsObjectNotSealed());
  VINEYARD_ASSERT(blob == nullptr);

  // unsafe get
  VINEYARD_CHECK_OK(client2.GetBlob(blob_writer->id(), true, blob));
  VINEYARD_ASSERT(blob != nullptr);
  CHECK_EQ(blob_writer->id(), blob->id());

  // shrink blob test
  VINEYARD_CHECK_OK(client1.CreateBlob(16 * 1024 * 1024, blob_writer));
  VINEYARD_CHECK_OK(blob_writer->Shrink(client1, 1024 * 1024));
  std::shared_ptr<Object> blob_object;
  VINEYARD_CHECK_OK(blob_writer->Seal(client1, blob_object));
  VINEYARD_ASSERT(blob_object != nullptr);

  // cannot be shrink after sealed
  CHECK(blob_writer->Shrink(client1, 1024 * 1024).IsObjectSealed());

  LOG(INFO) << "Passed various ways to get blob tests...";

  client1.Disconnect();
  client2.Disconnect();

  return 0;
}
