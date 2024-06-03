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

#include <string>

#include "basic/ds/tensor.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

void CreateBuffer(std::string& objectName, Client& client) {
  TensorBuilder<int> builder(client, {1, 10});
  int* data = builder.data();
  for (int i = 0; i < 10; ++i) {
    data[i] = i + 1000;
  }
  std::shared_ptr<Tensor<int>> tensor =
      std::dynamic_pointer_cast<Tensor<int>>(builder.Seal(client));
  client.Persist(tensor->id());
  LOG(INFO) << "Create object with id:"
            << reinterpret_cast<void*>(tensor->id());
  client.PutName(tensor->id(), objectName);
}

void FetchBuffer(std::string& objectName, Client& client) {
  ObjectID id;
  client.GetName(objectName, id);
  std::shared_ptr<Tensor<int>> tensor =
      std::dynamic_pointer_cast<Tensor<int>>(client.FetchAndGetObject(id));
  LOG(INFO) << "Fetch object with id:" << reinterpret_cast<void*>(tensor->id());
  const int* data = tensor->data();
  for (int i = 0; i < 10; ++i) {
    CHECK_EQ(data[i], i + 1000);
  }
}

int main(int argc, char** argv) {
  std::string sockets[2];
  Client client[2];
  if (argc < 2) {
    printf(
        "Usage ./fetch_remote_object_test "
        "<ipc_socket_1> <ipc_socket_2> -d \n");
    return 1;
  }

  for (int i = 0; i < 2; i++) {
    sockets[i] = std::string(argv[i + 1]);
    VINEYARD_CHECK_OK(client[i].Connect(sockets[i]));
  }

  std::string objectName = "test_tensor";
  CreateBuffer(objectName, client[0]);
  FetchBuffer(objectName, client[1]);

  LOG(INFO) << "Passed fetch and get object test!";
  return 0;
}
