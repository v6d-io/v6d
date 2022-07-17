/** Copyright 2020-2022 Alibaba Group Holding Limited.

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
#include <stdexcept>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/array.h"
#include "basic/ds/sequence.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

using namespace vineyard;  // NOLINT
using namespace std;       // NOLINT

double delta = 1E-10;

template <typename T>
ObjectID GetObjectID(const std::shared_ptr<Array<T>>& sealed_array) {
  return ObjectIDFromString(sealed_array->meta()
                                .MetaData()["buffer_"]["id"]
                                .template get_ref<std::string const&>());
}

template <typename T>
vector<T> InitArray(int size, std::function<T(int n)> init_func) {
  std::vector<T> array;
  array.resize(size);
  for (int i = 0; i < size; i++) {
    array[i] = init_func(i);
  }
  return array;
}

void BasicTest(Client& client) {
  auto double_array = InitArray<double>(250, [](int i) { return i; });
  ArrayBuilder<double> builder(client, double_array);
  auto sealed_double_array =
      std::dynamic_pointer_cast<Array<double>>(builder.Seal(client));
  ObjectID id1 = sealed_double_array->id();
  auto blob_id = GetObjectID(sealed_double_array);
  CHECK(blob_id != InvalidObjectID());
  CHECK(client.IsInUse(blob_id));

  // Here we need a CHECK_FATAL
  bool flag = false;
  try {
    ArrayBuilder<double> builder2(client, double_array);
    auto sealed_double_array2 =
        std::dynamic_pointer_cast<Array<double>>(builder2.Seal(client));
  } catch (std::runtime_error e) { flag = true; }
  CHECK(flag);
  VINEYARD_CHECK_OK(client.Release({id1, blob_id}));
  CHECK(!client.IsInUse(blob_id));
  ArrayBuilder<double> builder3(client, double_array);
  auto sealed_double_array3 =
      std::dynamic_pointer_cast<Array<double>>(builder3.Seal(client));
  auto blob_id2 = GetObjectID(sealed_double_array3);
  CHECK(client.IsSpilled(blob_id));
  CHECK(client.IsInUse(blob_id2));
  client.Disconnect();
  LOG(INFO) << "Finish Basic Test...";
}

void ReloadTest(Client& client) {
  auto double_array = InitArray<double>(10, [](int i) { return i; });
  auto string_array1 =
      InitArray<std::string>(50, [](int i) { return to_string(i + 100000); });
  auto string_array2 = InitArray<std::string>(
      50, [](int i) { return to_string(20 + i + 100000); });
  auto string_array3 = InitArray<std::string>(
      50, [](int i) { return to_string(40 + i + 100000); });
  ObjectID id, bid, id1, bid1, id2, bid2, id3, bid3;
  {
    ArrayBuilder<double> builder(client, double_array);
    auto sealed_double_array =
        std::dynamic_pointer_cast<Array<double>>(builder.Seal(client));
    id = sealed_double_array->id();
    bid = GetObjectID(sealed_double_array);
    CHECK(bid != InvalidObjectID());
    CHECK(client.IsInUse(bid));
    VINEYARD_CHECK_OK(client.Release({id, bid}));
  }
  {
    ArrayBuilder<std::string> builder(client, string_array1);
    auto sealed_string_array =
        std::dynamic_pointer_cast<Array<std::string>>(builder.Seal(client));
    id1 = sealed_string_array->id();
    bid1 = GetObjectID(sealed_string_array);
    CHECK(bid1 != InvalidObjectID());
    CHECK(client.IsInUse(bid1));
    VINEYARD_CHECK_OK(client.Release({id1, bid1}));
  }
  {
    ArrayBuilder<std::string> builder(client, string_array2);
    auto sealed_string_array =
        std::dynamic_pointer_cast<Array<std::string>>(builder.Seal(client));
    id2 = sealed_string_array->id();
    bid2 = GetObjectID(sealed_string_array);
    CHECK(bid2 != InvalidObjectID());
    CHECK(client.IsInUse(bid2));
    VINEYARD_CHECK_OK(client.Release({id2, bid2}));
  }
  {
    ArrayBuilder<std::string> builder(client, string_array3);
    auto sealed_string_array =
        std::dynamic_pointer_cast<Array<std::string>>(builder.Seal(client));
    id3 = sealed_string_array->id();
    bid3 = GetObjectID(sealed_string_array);
    CHECK(bid3 != InvalidObjectID());
    CHECK(client.IsInUse(bid3));
    VINEYARD_CHECK_OK(client.Release({id3, bid3}));
  }
  // now check for double_array
  {
    CHECK(client.IsSpilled(bid));
    auto double_array_copy = client.GetObject<Array<double>>(id);
    CHECK(double_array_copy->size() == double_array.size());
    for (size_t i = 0; i < double_array.size(); i++) {
      CHECK(abs(double_array[i] - (*double_array_copy)[i]) < delta);
    }
  }
  {
    CHECK(client.IsSpilled(bid1));
    auto str_array_copy = client.GetObject<Array<std::string>>(id1);
    CHECK(str_array_copy->size() == string_array1.size());
    for (size_t i = 0; i < string_array1.size(); i++) {
      CHECK_EQ(string_array1[i], (*str_array_copy)[i]);
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./spill_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client1, client2;
  VINEYARD_CHECK_OK(client1.Connect(ipc_socket));
  VINEYARD_CHECK_OK(client2.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  BasicTest(client1);
  ReloadTest(client2);

  LOG(INFO) << "Passed spill tests ...";

  client1.Disconnect();
  client2.Disconnect();
}
