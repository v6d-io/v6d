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

#include "client/client.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./name_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  ObjectID id = GenerateObjectID();
  VINEYARD_CHECK_OK(client.PutName(id, "test_name"));

  ObjectID id2 = 0;
  VINEYARD_CHECK_OK(client.GetName("test_name", id2));
  CHECK_EQ(id, id2);

  LOG(INFO) << "check existing name success";

  ObjectID id3 = 0;
  auto status = client.GetName("test_name2", id3);
  CHECK(status.IsObjectNotExists());

  LOG(INFO) << "check non-existing name success";

  VINEYARD_CHECK_OK(client.DropName("test_name"));
  ObjectID id4 = 0;
  auto status1 = client.GetName("test_name", id4);
  CHECK(status1.IsObjectNotExists());

  LOG(INFO) << "check drop name success";

  LOG(INFO) << "Passed name test...";

  client.Disconnect();

  return 0;
}
