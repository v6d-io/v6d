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

#include <unistd.h>
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

constexpr double delta = 1E-10;

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
  {
    bool is_in_use{false};
    VINEYARD_CHECK_OK(client.IsInUse(blob_id, is_in_use));
    CHECK(is_in_use);
  }

  {
    std::unique_ptr<BlobWriter> buffer_writer_;
    auto status =
        client.CreateBlob(double_array.size() * sizeof(double), buffer_writer_);
    CHECK(status.IsNotEnoughMemory());
  }

  bool is_spilled{false};
  bool is_in_use{false};
  VINEYARD_CHECK_OK(client.Release({id1, blob_id}));
  VINEYARD_CHECK_OK(client.IsInUse(blob_id, is_in_use));
  CHECK(!is_in_use);
  ArrayBuilder<double> builder3(client, double_array);
  auto sealed_double_array3 =
      std::dynamic_pointer_cast<Array<double>>(builder3.Seal(client));
  auto id2 = sealed_double_array3->id();
  auto blob_id2 = GetObjectID(sealed_double_array3);
  VINEYARD_CHECK_OK(client.IsSpilled(blob_id, is_spilled));
  CHECK(is_spilled);
  VINEYARD_CHECK_OK(client.IsInUse(blob_id2, is_in_use));
  CHECK(is_in_use);
  VINEYARD_CHECK_OK(client.Release({id2, blob_id2}));
  LOG(INFO) << "Finish Basic Test";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./spill_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client1;
  VINEYARD_CHECK_OK(client1.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  BasicTest(client1);

  LOG(INFO) << "Passed" << flush;

  client1.Disconnect();

  // avoid CrashLoopBackOff
  sleep(600);
}
