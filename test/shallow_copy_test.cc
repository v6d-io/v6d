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
    printf("usage ./shallow_copy_test <ipc_socket>");
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
  VINEYARD_CHECK_OK(client.Persist(id));

  ObjectID target_id = InvalidObjectID();
  VINEYARD_CHECK_OK(client.ShallowCopy(id, target_id));
  CHECK(target_id != InvalidObjectID());

  auto copied_vec =
      std::dynamic_pointer_cast<Array<double>>(client.GetObject(target_id));

  CHECK_EQ(sealed_double_array->size(), double_array.size());
  CHECK_EQ(copied_vec->size(), double_array.size());
  for (size_t i = 0; i < double_array.size(); ++i) {
    CHECK_EQ((*sealed_double_array)[i], double_array[i]);
    CHECK_EQ((*copied_vec)[i], double_array[i]);
  }

  VINEYARD_CHECK_OK(client.DelData(target_id));
  VINEYARD_CHECK_OK(client.DelData(id));

  LOG(INFO) << "Passed shallow copy tests...";

  client.Disconnect();

  return 0;
}
