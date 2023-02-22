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
    printf("usage ./object_meta_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  uint64_t t0 = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();
  std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};
  ArrayBuilder<double> builder(client, double_array);
  auto sealed_double_array =
      std::dynamic_pointer_cast<Array<double>>(builder.Seal(client));
  uint64_t t1 = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();

  ObjectID id = sealed_double_array->id();
  LOG(INFO) << "successfully sealed, " << ObjectIDToString(id) << " ...";

  {
    // check object meta's timestamp
    ObjectMeta meta;
    VINEYARD_CHECK_OK(client.GetMetaData(id, meta));
    CHECK_LE(t0, meta.Timestamp());
    CHECK_GE(t1, meta.Timestamp());
    LOG(INFO) << "Passed object_meta timestamp tests ...";
  }

  {
    // check object meta's labels
    ObjectMeta meta;
    VINEYARD_CHECK_OK(client.GetMetaData(id, meta));
    CHECK_EQ(0, meta.Labels().size());

    VINEYARD_CHECK_OK(client.Label(id, "label1", "value1"));
    VINEYARD_CHECK_OK(client.GetMetaData(id, meta));
    CHECK_EQ(1, meta.Labels().size());
    CHECK_EQ("value1", meta.Label("label1"));

    VINEYARD_CHECK_OK(client.Label(id, "label2", "value2"));
    VINEYARD_CHECK_OK(client.GetMetaData(id, meta));
    CHECK_EQ(2, meta.Labels().size());
    CHECK_EQ("value2", meta.Label("label2"));

    // override
    VINEYARD_CHECK_OK(client.Label(id, "label1", "value3"));
    VINEYARD_CHECK_OK(client.GetMetaData(id, meta));
    CHECK_EQ(2, meta.Labels().size());
    CHECK_EQ("value3", meta.Label("label1"));
  }

  LOG(INFO) << "Passed object_meta tests ...";

  client.Disconnect();

  return 0;
}
