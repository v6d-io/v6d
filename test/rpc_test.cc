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

#include <memory>
#include <string>
#include <thread>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/array.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "client/rpc_client.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage ./vector_test <ipc_socket> <rpc_endpoint>");
    return 1;
  }
  std::string ipc_socket(argv[1]);
  std::string rpc_endpoint(argv[2]);

  Client ipc_client;
  VINEYARD_CHECK_OK(ipc_client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  RPCClient rpc_client;
  VINEYARD_CHECK_OK(rpc_client.Connect(rpc_endpoint));
  LOG(INFO) << "Connected to RPCServer: " << rpc_endpoint;

  std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};

  ArrayBuilder<double> builder(ipc_client, double_array);
  auto sealed_double_array =
      std::dynamic_pointer_cast<Array<double>>(builder.Seal(ipc_client));
  VINEYARD_CHECK_OK(ipc_client.Persist(sealed_double_array->id()));

  ObjectID id = sealed_double_array->id();

  auto vy_double_array =
      std::dynamic_pointer_cast<Array<double>>(rpc_client.GetObject(id));
  CHECK_EQ(vy_double_array->id(), id);
  CHECK_EQ(vy_double_array->size(), double_array.size());
  CHECK_EQ(vy_double_array->size(), sealed_double_array->size());

  LOG(INFO) << "Passed rpc client tests...";

  ipc_client.Disconnect();
  rpc_client.Disconnect();

  return 0;
}
