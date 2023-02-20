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

#include "basic/ds/scalar.h"
#include "client/client.h"
#include "common/util/logging.h"

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

  // put name on non-existing object id
  ObjectID id = GenerateObjectID();
  CHECK(client.PutName(id, "test_name").IsObjectNotExists());

  // blob cannot have name
  CHECK(client.PutName(EmptyBlobID(), "test_name").IsInvalid());

  // generate a valid object id
  {
    ScalarBuilder<int32_t> scalar_builder(client);
    scalar_builder.SetValue(1234);

    auto scalar =
        std::dynamic_pointer_cast<Scalar<int32_t>>(scalar_builder.Seal(client));
    id = scalar->id();

    ObjectMeta meta;
    VINEYARD_CHECK_OK(client.GetMetaData(id, meta));
  }

  // transient object cannot have name
  CHECK(client.PutName(id, "test_name").IsInvalid());

  VINEYARD_CHECK_OK(client.Persist(id));

  VINEYARD_CHECK_OK(client.PutName(id, "test_name"));

  ObjectID id2 = 0;
  VINEYARD_CHECK_OK(client.GetName("test_name", id2));
  CHECK_EQ(id, id2);

  LOG(INFO) << "check existing name succeed";

  {
    std::map<std::string, ObjectID> names;
    VINEYARD_CHECK_OK(client.ListNames("test*", false, 100, names));
    CHECK(names.find("test_name") != names.end());
    CHECK(names.at("test_name") == id);
  }

  LOG(INFO) << "check list names succeed";

  // meta should contains name
  ObjectMeta meta;
  VINEYARD_CHECK_OK(client.GetMetaData(id, meta));
  CHECK(meta.HasKey("__name"));
  CHECK_EQ(meta.GetKeyValue<std::string>("__name"), "test_name");

  ObjectID id3 = 0;
  auto status = client.GetName("test_name2", id3);
  CHECK(status.IsObjectNotExists());

  LOG(INFO) << "check non-existing name succeed";

  VINEYARD_CHECK_OK(client.DropName("test_name"));
  ObjectID id4 = 0;
  CHECK(client.GetName("test_name", id4).IsObjectNotExists());

  LOG(INFO) << "check drop name succeed";

  // meta shouldn't contains name anymore
  VINEYARD_CHECK_OK(client.GetMetaData(id, meta));
  CHECK(!meta.HasKey("__name"));

  {
    std::map<std::string, ObjectID> names;
    VINEYARD_CHECK_OK(client.ListNames("test*", false, 100, names));
    // name being deleted
    CHECK(names.find("test_name") == names.end());
  }

  LOG(INFO) << "check list names succeed";

  LOG(INFO) << "Passed name test...";

  client.Disconnect();

  return 0;
}
