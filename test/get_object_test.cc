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
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./get_object_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};
  ArrayBuilder<double> builder(client, double_array);
  auto sealed_double_array =
      std::dynamic_pointer_cast<Array<double>>(builder.Seal(client));
  ObjectID id = sealed_double_array->id();
  ObjectID copied_id = InvalidObjectID();
  VINEYARD_CHECK_OK(client.ShallowCopy(id, copied_id));
  CHECK(copied_id != InvalidObjectID());

  {
    auto array1 =
        std::dynamic_pointer_cast<Array<double>>(client.GetObject(id));
    CHECK(array1 != nullptr);

    // mismatch type
    auto array2 = std::dynamic_pointer_cast<Array<float>>(client.GetObject(id));
    CHECK(array2 == nullptr);
  }

  {
    auto array1 = client.GetObject<Array<double>>(id);
    CHECK(array1 != nullptr);

    // mismatch type
    auto array2 = client.GetObject<Array<float>>(id);
    CHECK(array2 == nullptr);
  }

  {
    std::shared_ptr<Array<double>> array1;
    VINEYARD_CHECK_OK(client.GetObject(id, array1));
    CHECK(array1 != nullptr);

    // mismatch type
    std::shared_ptr<Array<float>> array2;
    auto status = client.GetObject(id, array2);
    CHECK(status.IsObjectNotExists());
    CHECK(array2 == nullptr);
  }

  auto arrays = client.GetObjects({id, copied_id});
  CHECK_EQ(arrays.size(), 2);
  CHECK_EQ(arrays[0]->id(), id);
  CHECK_EQ(arrays[1]->id(), copied_id);

  LOG(INFO) << "Passed various ways to get object tests...";

  client.Disconnect();

  return 0;
}
