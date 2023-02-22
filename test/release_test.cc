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

#if defined(__APPLE__) && defined(__clang__)
#define private public
#define protected public
#endif

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/array.h"
#include "basic/ds/sequence.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./release_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client1, client2;
  VINEYARD_CHECK_OK(client1.Connect(ipc_socket));
  VINEYARD_CHECK_OK(client2.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  ObjectID id = InvalidObjectID(), blob_id = InvalidObjectID(),
           copy_id = InvalidObjectID();

  {
    // prepare data
    std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};
    ArrayBuilder<double> builder(client1, double_array);
    std::shared_ptr<Object> sealed_double_array;
    VINEYARD_CHECK_OK(builder.Seal(client1, sealed_double_array));
    id = sealed_double_array->id();
    blob_id = ObjectIDFromString(sealed_double_array->meta()
                                     .MetaData()["buffer_"]["id"]
                                     .get_ref<std::string const&>());
    CHECK(blob_id != InvalidObjectID());
  }

  {  // basic
    bool is_in_use{false};
    VINEYARD_CHECK_OK(client1.IsInUse(blob_id, is_in_use));
    CHECK(is_in_use);
    VINEYARD_CHECK_OK(client1.Release({id, blob_id}));
    VINEYARD_CHECK_OK(client1.IsInUse(blob_id, is_in_use));
    CHECK(!is_in_use);
  }

  {  // single client
    bool is_in_use{false};
    auto obj = client1.GetObject(id);
    CHECK(obj != nullptr);
    VINEYARD_CHECK_OK(client1.IsInUse(blob_id, is_in_use));
    CHECK(is_in_use);
    auto blob = client1.GetObject(blob_id);
    CHECK(blob != nullptr);

    VINEYARD_CHECK_OK(client1.Release({id}));
    VINEYARD_CHECK_OK(client1.IsInUse(blob_id, is_in_use));
    CHECK(is_in_use);
    VINEYARD_CHECK_OK(client1.Release({blob_id}));
    VINEYARD_CHECK_OK(client1.IsInUse(blob_id, is_in_use));
    CHECK(!is_in_use);
  }

  {  // multiple clients
    auto blob1 = client1.GetObject(blob_id);
    CHECK(blob1 != nullptr);
    auto blob2 = client2.GetObject(blob_id);
    CHECK(blob2 != nullptr);
    VINEYARD_CHECK_OK(client1.Release({blob_id}));
    bool is_in_use{false};
    VINEYARD_CHECK_OK(client1.IsInUse(blob_id, is_in_use));
    CHECK(is_in_use);
    VINEYARD_CHECK_OK(client2.Release({blob_id}));
    VINEYARD_CHECK_OK(client2.IsInUse(blob_id, is_in_use));
    CHECK(!is_in_use);
  }

  {  // diamond reference count
    VINEYARD_CHECK_OK(client1.ShallowCopy(id, copy_id));
    auto obj1 = client1.GetObject(id);
    auto obj2 = client1.GetObject(copy_id);
    CHECK(obj1 != obj2);
    SequenceBuilder pair_builder1(client1);
    pair_builder1.SetSize(2);
    pair_builder1.SetValue(0, obj1);
    pair_builder1.SetValue(1, obj2);
    auto wrapper = pair_builder1.Seal(client1);
    CHECK(wrapper != nullptr);
    bool is_in_use{false};
    VINEYARD_CHECK_OK(client1.Release({wrapper->id()}));
    VINEYARD_CHECK_OK(client1.IsInUse(blob_id, is_in_use));
    CHECK(is_in_use);
    VINEYARD_CHECK_OK(client1.Release({id}));
    VINEYARD_CHECK_OK(client1.IsInUse(blob_id, is_in_use));
    CHECK(is_in_use);
    VINEYARD_CHECK_OK(client1.Release({copy_id}));
    VINEYARD_CHECK_OK(client1.IsInUse(blob_id, is_in_use));
    CHECK(!is_in_use);
  }

  {  // auto release in disconnection
    auto obj = client1.GetObject(id);
    CHECK(obj != nullptr);
    bool is_in_use{false};
    VINEYARD_CHECK_OK(client2.IsInUse(blob_id, is_in_use));
    CHECK(is_in_use);
    client1.Disconnect();
    sleep(5);
    VINEYARD_CHECK_OK(client2.IsInUse(blob_id, is_in_use));
    CHECK(!is_in_use);
  }

  LOG(INFO) << "Passed release tests...";

  client2.Disconnect();

  return 0;
}
