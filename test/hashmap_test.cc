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
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./hashmap_test <ipc_socket_name>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  HashmapBuilder<int, double> builder(client);
  builder[1] = 100.0;
  builder[2] = 50.0;
  builder[3] = 25.0;
  builder[4] = 12.5;
  builder[5] = 6.25;

  auto sealed_hashmap =
      std::dynamic_pointer_cast<Hashmap<int, double>>(builder.Seal(client));
  CHECK(!sealed_hashmap->IsPersist());
  CHECK(sealed_hashmap->IsLocal());
  VINEYARD_CHECK_OK(sealed_hashmap->Persist(client));
  CHECK(sealed_hashmap->IsPersist());
  CHECK(sealed_hashmap->IsLocal());

  ObjectID id = sealed_hashmap->id();
  LOG(INFO) << "hashmap id: " << id;

  auto vy_hashmap =
      std::dynamic_pointer_cast<Hashmap<int, double>>(client.GetObject(id));

  CHECK_EQ(builder.size(), sealed_hashmap->size());
  CHECK_EQ(builder.size(), vy_hashmap->size());
  LOG(INFO) << "after check size...";

  for (const auto& pair : builder) {
    CHECK_DOUBLE_EQ(pair.second, sealed_hashmap->at(pair.first));
    CHECK_DOUBLE_EQ(pair.second, vy_hashmap->at(pair.first));
  }

  LOG(INFO) << "Passed double hashmap tests...";

  client.Disconnect();

  return 0;
}
