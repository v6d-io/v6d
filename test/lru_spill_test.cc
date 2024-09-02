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
#include <vector>

#include "basic/ds/array.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "client/rpc_client.h"
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

/**
 * @brief Test whether the LRU mechanism of spill works.
 *
 * @note Suppose the vineyard server memory is 8000 bytes,
 *       the spill_high_watermark is 0.8, and the spill_low_watermark is 0.5.
 *       In this test, we create 10 arrays, each of which has 250 doubles.
 *       So the vineyardd server can only hold 3 arrays at most, when it
 *       reaches the high watermark, it will spill the least recently used
 *       3 arrays to the disk.
 *       The test steps are as follows:
 *       1. IPCClient create: Array1, Array2 => (Array1, Array2)
 *       2. RPCClient create: Array3, Array4 => (Array1, Array2, Array3, Array4)
 * => Spill Array1, Array2. => (Array3, Array4)
 *       3. RPCClient get: Array2. => (Array3, Array4, Array2)
 *       4. IPCClient create: Array5 => (Array3, Array4, Array2, Array5) =>
 * Spill Array3, Array4. => (Array2, Array5)
 *       5. IPCClient get: Array2 => (Array5, Array2)
 *       6. IPCClient get: Array4 => (Array5, Array2, Array4)
 *       7. RPCClient create: Array6 => (Array5, Array2, Array4, Array6) =>
 * Spill Array5, Array2. => (Array4, Array6)
 *       8. RPCClient get: Array1 => (Array4, Array6, Array1)
 *       9. RPCClient get: Array3 => (Array4, Array6, Array1, Array3) => Spill
 * Array4, Array6 => (Array1, Array3)
 *       10. RPCClient get: Array6 => (Array1, Array3, Array6)
 *       11. IPCClient get: Array3, Array4 => (Array1, Array6, Array3, Array4)
 * => Spill Array1, Array6. => (Array3, Array4)
 *       12. RPCClient create: Array7 => (Array3, Array4, Array7)
 *       13. IPCClient create: Array8, Array9, Array10 => (Array3, Array4,
 * Array7, Array8) => Spill Array3, Array4. => (Array7, Array8) => (Array7,
 * Array8, Array9, Array10) => Spill Array7, Array8. => (Array9, Array10)
 *       14. RPCClient get: Array5, Array3, Array1 => (Array9, Array10, Array5,
 * Array3) => Spill Array9, Array10. => (Array5, Array3, Array1)
 *       15. IPCClient get: Array2, Array4, Array6 => (Array5, Array3, Array1,
 * Array2) => Spill Array5, Array3. => (Array1, Array2, Array4, Array6) => Spill
 * Array1, Array2. => (Array4, Array6)
 *
 */

void LRUTest(Client& client, RPCClient& rpc_client) {
  auto double_array = InitArray<double>(250, [](int i) { return i; });
  /* step1: IPCClient create: Array1, Array2 => (Array1, Array2) */
  ArrayBuilder<double> builder1(client, double_array);
  auto sealed_double_array1 =
      std::dynamic_pointer_cast<Array<double>>(builder1.Seal(client));
  VINEYARD_CHECK_OK(client.Release(sealed_double_array1->id()));

  ArrayBuilder<double> builder2(client, double_array);
  auto sealed_double_array2 =
      std::dynamic_pointer_cast<Array<double>>(builder2.Seal(client));
  VINEYARD_CHECK_OK(client.Release(sealed_double_array2->id()));

  /* step2: RPCClient create: Array3, Array4 */
  auto remote_blob_writer3 =
      std::make_shared<RemoteBlobWriter>(double_array.size() * sizeof(double));
  std::memcpy(remote_blob_writer3->data(), double_array.data(),
              double_array.size() * sizeof(double));
  ObjectMeta blob_meta3;
  VINEYARD_CHECK_OK(
      rpc_client.CreateRemoteBlob(remote_blob_writer3, blob_meta3));

  auto remote_blob_writer4 =
      std::make_shared<RemoteBlobWriter>(double_array.size() * sizeof(double));
  std::memcpy(remote_blob_writer4->data(), double_array.data(),
              double_array.size() * sizeof(double));
  ObjectMeta blob_meta4;
  VINEYARD_CHECK_OK(
      rpc_client.CreateRemoteBlob(remote_blob_writer4, blob_meta4));

  //  (Array1, Array2, Array3, Array4) => Spill Array1, Array2. => (Array3,
  //  Array4)
  bool is_spilled{false};
  VINEYARD_CHECK_OK(
      client.IsSpilled(GetObjectID(sealed_double_array1), is_spilled));
  CHECK(is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(
      client.IsSpilled(GetObjectID(sealed_double_array2), is_spilled));
  CHECK(is_spilled);

  /* step3: RPCClient get: Array2. => (Array3, Array4, Array2) */
  std::shared_ptr<Object> object;
  VINEYARD_CHECK_OK(rpc_client.GetObject(sealed_double_array2->id(), object));

  /* step4: IPCClient create: Array5 */
  ArrayBuilder<double> builder5(client, double_array);
  auto sealed_double_array5 =
      std::dynamic_pointer_cast<Array<double>>(builder5.Seal(client));
  VINEYARD_CHECK_OK(client.Release(sealed_double_array5->id()));

  // (Array3, Array4, Array2, Array5) => Spill Array3, Array4. => (Array2,
  // Array5)
  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta3.GetId(), is_spilled));
  CHECK(is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta4.GetId(), is_spilled));
  CHECK(is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(
      client.IsSpilled(GetObjectID(sealed_double_array5), is_spilled));
  CHECK(!is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(
      client.IsSpilled(GetObjectID(sealed_double_array2), is_spilled));
  CHECK(!is_spilled);

  /* step5: IPCClient get: Array2 => (Array5, Array2) */
  auto obj = client.GetObject(sealed_double_array2->id());
  VINEYARD_CHECK_OK(client.Release(sealed_double_array2->id()));
  CHECK(obj != nullptr);

  /* step6: IPCClient get: Array4 => (Array5, Array2, Array4) */
  obj = client.GetObject(blob_meta4.GetId());
  VINEYARD_CHECK_OK(client.Release(blob_meta4.GetId()));
  CHECK(obj != nullptr);

  /* step7: RPCClient create: Array6 */
  auto remote_blob_writer6 =
      std::make_shared<RemoteBlobWriter>(double_array.size() * sizeof(double));
  std::memcpy(remote_blob_writer6->data(), double_array.data(),
              double_array.size() * sizeof(double));
  ObjectMeta blob_meta6;
  VINEYARD_CHECK_OK(
      rpc_client.CreateRemoteBlob(remote_blob_writer6, blob_meta6));

  // (Array5, Array2, Array4, Array6) => Spill Array5, Array2. => (Array4,
  // Array6)
  is_spilled = false;
  VINEYARD_CHECK_OK(
      client.IsSpilled(GetObjectID(sealed_double_array5), is_spilled));
  CHECK(is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(
      client.IsSpilled(GetObjectID(sealed_double_array2), is_spilled));
  CHECK(is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta4.GetId(), is_spilled));
  CHECK(!is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta6.GetId(), is_spilled));
  CHECK(!is_spilled);

  /* step8: RPCClient get: Array1 => (Array4, Array6, Array1) */
  VINEYARD_CHECK_OK(rpc_client.GetObject(sealed_double_array1->id(), object));

  /* step9: RPCClient get: Array3 */
  VINEYARD_CHECK_OK(rpc_client.GetObject(blob_meta3.GetId(), object));

  // (Array4, Array6, Array1, Array3) => Spill Array4, Array6 => (Array1,
  // Array3)
  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta4.GetId(), is_spilled));
  CHECK(is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta6.GetId(), is_spilled));
  CHECK(is_spilled);

  /* step10: RPCClient get: Array6 => (Array1, Array3, Array6) */
  VINEYARD_CHECK_OK(rpc_client.GetObject(blob_meta6.GetId(), object));

  /* step11: IPCClient get: Array3, Array4 */
  obj = client.GetObject(blob_meta3.GetId());
  VINEYARD_CHECK_OK(client.Release(blob_meta3.GetId()));
  CHECK(obj != nullptr);

  obj = client.GetObject(blob_meta4.GetId());
  VINEYARD_CHECK_OK(client.Release(blob_meta4.GetId()));
  CHECK(obj != nullptr);

  // (Array1, Array6, Array3, Array4) => Spill Array1, Array6. => (Array3,
  // Array4)
  is_spilled = false;
  VINEYARD_CHECK_OK(
      client.IsSpilled(GetObjectID(sealed_double_array1), is_spilled));
  CHECK(is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta6.GetId(), is_spilled));
  CHECK(is_spilled);

  /* step12: RPCClient create: Array7 */
  auto remote_blob_writer7 =
      std::make_shared<RemoteBlobWriter>(double_array.size() * sizeof(double));
  std::memcpy(remote_blob_writer7->data(), double_array.data(),
              double_array.size() * sizeof(double));
  ObjectMeta blob_meta7;
  VINEYARD_CHECK_OK(
      rpc_client.CreateRemoteBlob(remote_blob_writer7, blob_meta7));

  // (Array3, Array4) => (Array3, Array4, Array7)
  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta3.GetId(), is_spilled));
  CHECK(!is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta4.GetId(), is_spilled));
  CHECK(!is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta7.GetId(), is_spilled));
  CHECK(!is_spilled);

  /* step13: IPCClient create: Array8, Array9, Array10 */
  ArrayBuilder<double> builder8(client, double_array);
  auto sealed_double_array8 =
      std::dynamic_pointer_cast<Array<double>>(builder8.Seal(client));
  VINEYARD_CHECK_OK(client.Release(sealed_double_array8->id()));

  ArrayBuilder<double> builder9(client, double_array);
  auto sealed_double_array9 =
      std::dynamic_pointer_cast<Array<double>>(builder9.Seal(client));
  VINEYARD_CHECK_OK(client.Release(sealed_double_array9->id()));

  ArrayBuilder<double> builder10(client, double_array);
  auto sealed_double_array10 =
      std::dynamic_pointer_cast<Array<double>>(builder10.Seal(client));
  VINEYARD_CHECK_OK(client.Release(sealed_double_array10->id()));
  // (Array3, Array4, Array7, Array8) => Spill Array3, Array4. => (Array7,
  // Array8) (Array7, Array8, Array9, Array10) => Spill Array7, Array8. =>
  // (Array9, Array10)
  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta3.GetId(), is_spilled));
  CHECK(is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta4.GetId(), is_spilled));
  CHECK(is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta7.GetId(), is_spilled));
  CHECK(is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(
      client.IsSpilled(GetObjectID(sealed_double_array8), is_spilled));
  CHECK(is_spilled);

  /* step14: RPCClient get: Array5, Array3, Array1 */
  VINEYARD_CHECK_OK(rpc_client.GetObject(sealed_double_array5->id(), object));

  VINEYARD_CHECK_OK(rpc_client.GetObject(blob_meta3.GetId(), object));

  VINEYARD_CHECK_OK(rpc_client.GetObject(sealed_double_array1->id(), object));
  // (Array9, Array10, Array5, Array3) => Spill Array9, Array10 => (Array5,
  // Array3) (Array5, Array3) => (Array5, Array3, Array1)
  is_spilled = false;
  VINEYARD_CHECK_OK(
      client.IsSpilled(GetObjectID(sealed_double_array9), is_spilled));
  CHECK(is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(
      client.IsSpilled(GetObjectID(sealed_double_array10), is_spilled));
  CHECK(is_spilled);

  /* step15: IPCClient get: Array2, Array4, Array6 */
  obj = client.GetObject(sealed_double_array2->id());
  VINEYARD_CHECK_OK(client.Release(sealed_double_array2->id()));
  CHECK(obj != nullptr);

  obj = client.GetObject(blob_meta4.GetId());
  VINEYARD_CHECK_OK(client.Release(blob_meta4.GetId()));
  CHECK(obj != nullptr);

  obj = client.GetObject(blob_meta6.GetId());
  VINEYARD_CHECK_OK(client.Release(blob_meta6.GetId()));
  CHECK(obj != nullptr);

  // (Array5, Array3, Array1, Array2) => Spill Array5, Array3. => (Array1,
  // Array2)
  is_spilled = false;
  VINEYARD_CHECK_OK(
      client.IsSpilled(GetObjectID(sealed_double_array5), is_spilled));
  CHECK(is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(client.IsSpilled(blob_meta3.GetId(), is_spilled));
  CHECK(is_spilled);

  // (Array1, Array2, Array4, Array6) => Spill Array1, Array2. => (Array4,
  // Array6)
  is_spilled = false;
  VINEYARD_CHECK_OK(
      client.IsSpilled(GetObjectID(sealed_double_array1), is_spilled));
  CHECK(is_spilled);

  is_spilled = false;
  VINEYARD_CHECK_OK(
      client.IsSpilled(GetObjectID(sealed_double_array2), is_spilled));
  CHECK(is_spilled);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage ./lru_spill_test <ipc_socket> <rpc_endpoint>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);
  std::string rpc_endpoint = std::string(argv[2]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  RPCClient rpc_client;
  VINEYARD_CHECK_OK(rpc_client.Connect(rpc_endpoint));
  LOG(INFO) << "Connected to RPCServer: " << rpc_endpoint;

  LRUTest(client, rpc_client);
  VINEYARD_CHECK_OK(client.Clear());
  VINEYARD_CHECK_OK(rpc_client.Clear());

  client.Disconnect();
  rpc_client.Disconnect();

  LOG(INFO) << "Passed lru spill tests ...";
}
