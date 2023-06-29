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
#include "basic/ds/hashmap.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    LOG(INFO) << "usage ./perfect_hashmap_test <ipc_socket_name>";
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  PerfectHashmapBuilder<int, double> builder(client);
  std::vector<int> keys = {1, 2, 3, 4, 5};
  std::vector<double> values = {100.0, 50.0, 25.0, 12.5, 6.25};
  VINEYARD_CHECK_OK(
      builder.ComputeHash(client, keys.data(), values.data(), keys.size()));

  auto sealed_perfec_hashmap =
      std::dynamic_pointer_cast<PerfectHashmap<int, double>>(
          builder.Seal(client));
  CHECK(!sealed_perfec_hashmap->IsPersist());
  CHECK(sealed_perfec_hashmap->IsLocal());
  VINEYARD_CHECK_OK(sealed_perfec_hashmap->Persist(client));
  CHECK(sealed_perfec_hashmap->IsPersist());
  CHECK(sealed_perfec_hashmap->IsLocal());

  ObjectID id = sealed_perfec_hashmap->id();
  LOG(INFO) << "Perfect hashmap id: " << id;

  auto vy_hashmap = std::dynamic_pointer_cast<PerfectHashmap<int, double>>(
      client.GetObject(id));

  CHECK_EQ(builder.size(), sealed_perfec_hashmap->size());
  CHECK_EQ(builder.size(), vy_hashmap->size());
  LOG(INFO) << "after check size...";

  for (size_t i = 0; i < keys.size(); i++) {
    CHECK_DOUBLE_EQ(values[i], sealed_perfec_hashmap->at(keys[i]));
    CHECK_DOUBLE_EQ(values[i], vy_hashmap->at(keys[i]));
  }

  LOG(INFO) << "Passed double perfect hashmap tests...";

  client.Disconnect();

  return 0;
}
