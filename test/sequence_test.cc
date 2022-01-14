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

#include "arrow/status.h"
#include "arrow/util/io_util.h"
#include "arrow/util/logging.h"

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
    printf("usage ./sequence_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  SequenceBuilder tup_builder(client);
  tup_builder.SetSize(3);
  {
    std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};
    auto builder = std::make_shared<ArrayBuilder<double>>(client, double_array);
    tup_builder.SetValue(0, builder);
  }
  {
    auto builder = std::make_shared<HashmapBuilder<int, double>>(client);
    builder->operator[](1) = 100.0;
    builder->operator[](2) = 50.0;
    builder->operator[](3) = 25.0;
    builder->operator[](4) = 12.5;
    builder->operator[](5) = 6.25;
    tup_builder.SetValue(1, builder);
  }
  {
    std::vector<double> double_array = {9.0, 7.0, 5.0, 3.0, 1.0};
    auto builder = std::make_shared<ArrayBuilder<double>>(client, double_array);
    tup_builder.SetValue(2, builder);
  }

  auto tup = std::dynamic_pointer_cast<Sequence>(tup_builder.Seal(client));
  VINEYARD_CHECK_OK(client.Persist(tup->id()));

  auto first = tup->At(0);
  auto second = tup->At(1);
  auto third = tup->At(2);

  CHECK_EQ(first->meta().GetTypeName(), type_name<Array<double>>());
  {
    auto type = type_name<Hashmap<int, double>>();
    CHECK_EQ(second->meta().GetTypeName(), type);
  }
  CHECK_EQ(third->meta().GetTypeName(), type_name<Array<double>>());

  LOG(INFO) << "Passed tuple tests...";

  client.Disconnect();

  return 0;
}
