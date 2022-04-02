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
#include "basic/ds/hashmap.h"
#include "basic/ds/sequence.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"
#include "common/util/typename.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./large_meta_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  const size_t element_size = 10240;

  SequenceBuilder tup_builder(client);
  tup_builder.SetSize(element_size);
  for (size_t idx = 0; idx < element_size; ++idx) {
    std::vector<double> double_array = {1.0, static_cast<double>(element_size),
                                        static_cast<double>(idx)};
    auto builder = std::make_shared<ArrayBuilder<double>>(client, double_array);
    tup_builder.SetValue(idx, builder);
  }

  auto tup = std::dynamic_pointer_cast<Sequence>(tup_builder.Seal(client));
  VINEYARD_CHECK_OK(client.Persist(tup->id()));

  for (size_t idx = 0; idx < element_size; ++idx) {
    auto item = tup->At(idx);
    CHECK_EQ(item->meta().GetTypeName(), type_name<Array<double>>());
    auto arr = std::dynamic_pointer_cast<Array<double>>(item);
    CHECK_EQ(arr->size(), 3);
    CHECK_DOUBLE_EQ(arr->data()[0], 1.0);
    CHECK_DOUBLE_EQ(arr->data()[1], static_cast<double>(element_size));
    CHECK_DOUBLE_EQ(arr->data()[2], static_cast<double>(idx));
  }

  LOG(INFO) << "Passed large metadata tests...";

  client.Disconnect();

  return 0;
}
