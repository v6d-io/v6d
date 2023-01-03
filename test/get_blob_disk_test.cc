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

void testAnonymousDiskBlob(Client& client1, Client& client2) {
  std::unique_ptr<BlobWriter> blob_writer;
  VINEYARD_CHECK_OK(client1.CreateDiskBlob(1024, "", blob_writer));
  VINEYARD_ASSERT(blob_writer != nullptr);

  std::shared_ptr<Blob> blob;

  // safe get
  CHECK(client2.GetBlob(blob_writer->id(), blob).IsObjectNotSealed());
  VINEYARD_ASSERT(blob == nullptr);

  // unsafe get
  VINEYARD_CHECK_OK(client2.GetBlob(blob_writer->id(), true, blob));
  VINEYARD_ASSERT(blob != nullptr);
  CHECK_EQ(blob_writer->id(), blob->id());

  // write to blob
  for (size_t i = 0; i < 1024; ++i) {
    blob_writer->data()[i] = static_cast<int8_t>(i);
  }

  // msync
  CHECK_EQ(0, msync(blob_writer->data(), 1024, MS_SYNC));

  // compare the content of blob and blob writer
  for (size_t i = 0; i < 1024; ++i) {
    CHECK_EQ(blob_writer->data()[i], blob->data()[i]);
  }
}

void testNamedDiskBlob(Client& client1, Client& client2) {
  std::string path = "/tmp/test-named-blob";

  std::unique_ptr<BlobWriter> blob_writer;
  VINEYARD_CHECK_OK(client1.CreateDiskBlob(1024, path, blob_writer));
  VINEYARD_ASSERT(blob_writer != nullptr);

  std::shared_ptr<Blob> blob;

  // safe get
  CHECK(client2.GetBlob(blob_writer->id(), blob).IsObjectNotSealed());
  VINEYARD_ASSERT(blob == nullptr);

  // unsafe get
  VINEYARD_CHECK_OK(client2.GetBlob(blob_writer->id(), true, blob));
  VINEYARD_ASSERT(blob != nullptr);
  CHECK_EQ(blob_writer->id(), blob->id());

  // write to blob
  for (size_t i = 0; i < 1024; ++i) {
    blob_writer->data()[i] = static_cast<int8_t>(i);
  }

  // msync
  CHECK_EQ(0, msync(blob_writer->data(), 1024, MS_SYNC));

  // compare the content of blob and blob writer
  for (size_t i = 0; i < 1024; ++i) {
    CHECK_EQ(blob_writer->data()[i], blob->data()[i]);
  }

  // compare the content of blob and file on disk
  FILE* fileptr = fopen(path.c_str(), "rb");  // Open the file in binary mode
  fseek(fileptr, 0, SEEK_END);                // Jump to the end of the file
  size_t filelen = ftell(fileptr);  // Get the current byte offset in the file
  CHECK_EQ(filelen, 1024);

  rewind(fileptr);  // Jump back to the beginning of the file
  char* fbuffer = static_cast<char*>(
      malloc(filelen * sizeof(char)));  // Enough memory for the file
  CHECK_EQ(1, fread(fbuffer, filelen, 1, fileptr));  // Read in the entire file
  fclose(fileptr);                                   // Close the file

  for (size_t i = 0; i < 1024; ++i) {
    CHECK_EQ(blob_writer->data()[i], fbuffer[i]);
  }
  free(fbuffer);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./get_blob_disk_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client1;
  Client client2;
  VINEYARD_CHECK_OK(client1.Connect(ipc_socket));
  VINEYARD_CHECK_OK(client2.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  testAnonymousDiskBlob(client1, client2);
  testNamedDiskBlob(client1, client2);

  LOG(INFO) << "Passed various ways to get blob on disk tests...";

  client1.Disconnect();
  client2.Disconnect();

  return 0;
}
