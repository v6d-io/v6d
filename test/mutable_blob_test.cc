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
    printf("usage ./mutable_blob_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client1;
  VINEYARD_CHECK_OK(client1.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  Client client2;
  VINEYARD_CHECK_OK(client2.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  std::unique_ptr<BlobWriter> blob_writer;
  std::shared_ptr<Blob> blob;

  // create blob
  VINEYARD_CHECK_OK(client1.CreateBlob(1024, blob_writer));

  // cannot get by default
  CHECK(client2.GetBlob(blob_writer->id(), blob).IsObjectNotSealed());

  // cannot get as "safe"
  CHECK(client2.GetBlob(blob_writer->id(), false, blob).IsObjectNotSealed());

  // can get as "unsafe"
  VINEYARD_CHECK_OK(client2.GetBlob(blob_writer->id(), true, blob));
  CHECK_EQ(blob->id(), blob_writer->id());

  // seal blob
  blob_writer->Seal(client1);

  // can get by default after seal
  VINEYARD_CHECK_OK(client2.GetBlob(blob_writer->id(), blob));
  CHECK_EQ(blob->id(), blob_writer->id());

  // can get as "safe" after seal
  VINEYARD_CHECK_OK(client2.GetBlob(blob_writer->id(), false, blob));
  CHECK_EQ(blob->id(), blob_writer->id());

  // can get as "unsafe" after seal
  VINEYARD_CHECK_OK(client2.GetBlob(blob_writer->id(), true, blob));
  CHECK_EQ(blob->id(), blob_writer->id());

  LOG(INFO) << "Passed mutable blob test ...";

  client1.Disconnect();
  client2.Disconnect();

  return 0;
}
