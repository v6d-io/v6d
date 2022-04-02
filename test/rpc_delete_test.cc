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
#include "client/rpc_client.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage ./rpc_delete_test <ipc_socket> <rpc_endpoint>");
    return 1;
  }
  std::string ipc_socket(argv[1]);
  std::string rpc_endpoint(argv[2]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  RPCClient rpc_client;
  VINEYARD_CHECK_OK(rpc_client.Connect(rpc_endpoint));
  LOG(INFO) << "Connected to RPCServer: " << rpc_endpoint;

  ObjectID id = InvalidObjectID(), blob_id = InvalidObjectID(),
           wrapper_id = InvalidObjectID();
  bool exists;

  {
    // prepare data
    std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};
    ArrayBuilder<double> builder(client, double_array);
    auto sealed_double_array =
        std::dynamic_pointer_cast<Array<double>>(builder.Seal(client));
    id = sealed_double_array->id();
    blob_id = ObjectIDFromString(sealed_double_array->meta()
                                     .MetaData()["buffer_"]["id"]
                                     .get_ref<std::string const&>());
    CHECK(blob_id != InvalidObjectID());
  }

  {
    std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
    auto s = client.GetBuffers({blob_id}, buffers);
    CHECK(s.ok() && buffers.size() == 1);
  }

  // delete transient object
  VINEYARD_CHECK_OK(rpc_client.Exists(id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(blob_id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.DelData(id, false, true));
  VINEYARD_CHECK_OK(rpc_client.Exists(id, exists));
  CHECK(!exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(blob_id, exists));
  CHECK(!exists);

  {
    std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
    auto s = client.GetBuffers({blob_id}, buffers);
    CHECK(s.ok() && buffers.size() == 0);
  }

  {
    // prepare data
    std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};
    ArrayBuilder<double> builder(client, double_array);
    auto sealed_double_array =
        std::dynamic_pointer_cast<Array<double>>(builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(sealed_double_array->id()));
    id = sealed_double_array->id();
    blob_id = ObjectIDFromString(sealed_double_array->meta()
                                     .MetaData()["buffer_"]["id"]
                                     .get_ref<std::string const&>());
    CHECK(blob_id != InvalidObjectID());
  }

  {
    std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
    auto s = client.GetBuffers({blob_id}, buffers);
    CHECK(s.ok() && buffers.size() == 1);
  }

  // deep deletion
  VINEYARD_CHECK_OK(rpc_client.Exists(id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(blob_id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.DelData(id, false, true));
  VINEYARD_CHECK_OK(rpc_client.Exists(id, exists));
  CHECK(!exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(blob_id, exists));
  CHECK(!exists);

  // the blob should have been removed
  {
    std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
    auto s = client.GetBuffers({blob_id}, buffers);
    CHECK(s.ok() && buffers.size() == 0);
  }

  {
    // prepare data
    std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};
    ArrayBuilder<double> builder(client, double_array);
    auto sealed_double_array =
        std::dynamic_pointer_cast<Array<double>>(builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(sealed_double_array->id()));
    id = sealed_double_array->id();
    blob_id = ObjectIDFromString(sealed_double_array->meta()
                                     .MetaData()["buffer_"]["id"]
                                     .get_ref<std::string const&>());
    CHECK(blob_id != InvalidObjectID());
  }

  {
    std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
    auto s = client.GetBuffers({blob_id}, buffers);
    CHECK(s.ok() && buffers.size() == 1);
  }

  // shallow deletion
  VINEYARD_CHECK_OK(rpc_client.Exists(id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(blob_id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.DelData(id, false, false));
  VINEYARD_CHECK_OK(rpc_client.Exists(id, exists));
  CHECK(!exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(blob_id, exists));
  CHECK(!exists);  // see Note [Deleting objects and blobs]

  {
    std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
    auto s = client.GetBuffers({blob_id}, buffers);
    // the deletion on direct blob member is not shallow
    CHECK(s.ok() && buffers.size() == 0);
  }

  {
    // prepare data
    std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};
    ArrayBuilder<double> builder(client, double_array);
    auto sealed_double_array =
        std::dynamic_pointer_cast<Array<double>>(builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(sealed_double_array->id()));
    id = sealed_double_array->id();
    blob_id = ObjectIDFromString(sealed_double_array->meta()
                                     .MetaData()["buffer_"]["id"]
                                     .get_ref<std::string const&>());
    CHECK(blob_id != InvalidObjectID());

    // wrap
    SequenceBuilder pair_builder(client);
    pair_builder.SetSize(2);
    pair_builder.SetValue(0, sealed_double_array);
    pair_builder.SetValue(1, sealed_double_array);
    wrapper_id = pair_builder.Seal(client)->id();
  }

  {
    std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
    auto s = client.GetBuffers({blob_id}, buffers);
    CHECK(s.ok() && buffers.size() == 1);
  }

  // shallow deletion on object members
  VINEYARD_CHECK_OK(rpc_client.Exists(wrapper_id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(blob_id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.DelData(wrapper_id, false, false));
  VINEYARD_CHECK_OK(rpc_client.Exists(wrapper_id, exists));
  CHECK(!exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(blob_id, exists));
  CHECK(exists);  // see Note [Deleting objects and blobs]

  {
    std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
    auto s = client.GetBuffers({blob_id}, buffers);
    // the deletion on non-direct blob member is shallow
    CHECK(s.ok() && buffers.size() == 1);
  }

  {
    // prepare data
    std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};
    ArrayBuilder<double> builder(client, double_array);
    auto sealed_double_array =
        std::dynamic_pointer_cast<Array<double>>(builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(sealed_double_array->id()));
    id = sealed_double_array->id();
    blob_id = sealed_double_array->meta().GetMemberMeta("buffer_").GetId();
    CHECK(blob_id != InvalidObjectID());
  }

  // force deletion
  VINEYARD_CHECK_OK(rpc_client.Exists(id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(blob_id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.DelData(blob_id, true, false));
  VINEYARD_CHECK_OK(rpc_client.Exists(id, exists));
  CHECK(!exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(blob_id, exists));
  CHECK(!exists);

  // the blob should have been removed
  {
    std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
    auto s = client.GetBuffers({blob_id}, buffers);
    CHECK(s.ok() && buffers.size() == 0);
  }

  {
    // prepare data
    std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};
    ArrayBuilder<double> builder(client, double_array);
    auto sealed_double_array =
        std::dynamic_pointer_cast<Array<double>>(builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(sealed_double_array->id()));
    id = sealed_double_array->id();
    blob_id = sealed_double_array->meta().GetMemberMeta("buffer_").GetId();
    CHECK(blob_id != InvalidObjectID());
  }

  // shallow delete multiple objects
  VINEYARD_CHECK_OK(rpc_client.Exists(id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(blob_id, exists));
  CHECK(exists);
  VINEYARD_CHECK_OK(rpc_client.DelData({id, blob_id}, false, false));
  VINEYARD_CHECK_OK(rpc_client.Exists(id, exists));
  CHECK(!exists);
  VINEYARD_CHECK_OK(rpc_client.Exists(blob_id, exists));
  CHECK(!exists);

  // the blob should have been removed
  {
    std::map<ObjectID, std::shared_ptr<arrow::Buffer>> buffers;
    auto s = client.GetBuffers({blob_id}, buffers);
    CHECK(s.ok() && buffers.size() == 0);
  }

  // delete on complex data: and empty blob is quite special, since it cannot
  // been truely deleted.
  std::shared_ptr<InstanceStatus> status_before;
  VINEYARD_CHECK_OK(rpc_client.InstanceStatus(status_before));

  // prepare data
  ObjectID nested_tuple_id;
  {
    // prepare data
    std::vector<double> double_array1 = {1.0, 7.0, 3.0, 4.0, 2.0};
    ArrayBuilder<double> builder1(client, double_array1);

    std::vector<double> double_array2 = {1.0, 7.0, 3.0, 4.0, 2.0};
    ArrayBuilder<double> builder2(client, double_array2);

    std::vector<double> double_array3 = {1.0, 7.0, 3.0, 4.0, 2.0};
    ArrayBuilder<double> builder3(client, double_array3);

    std::vector<double> double_array4 = {1.0, 7.0, 3.0, 4.0, 2.0};
    ArrayBuilder<double> builder4(client, double_array4);

    SequenceBuilder pair_builder1(client);
    pair_builder1.SetSize(2);
    pair_builder1.SetValue(0, builder1.Seal(client));
    pair_builder1.SetValue(1, builder2.Seal(client));

    SequenceBuilder pair_builder2(client);
    pair_builder2.SetSize(2);
    pair_builder2.SetValue(0, builder3.Seal(client));
    pair_builder2.SetValue(1, Blob::MakeEmpty(client));

    SequenceBuilder pair_builder3(client);
    pair_builder3.SetSize(2);
    pair_builder3.SetValue(0, Blob::MakeEmpty(client));
    pair_builder3.SetValue(1, builder4.Seal(client));

    SequenceBuilder tuple_builder(client);
    tuple_builder.SetSize(3);
    tuple_builder.SetValue(0, pair_builder1.Seal(client));
    tuple_builder.SetValue(1, pair_builder2.Seal(client));
    tuple_builder.SetValue(2, pair_builder3.Seal(client));
    nested_tuple_id = tuple_builder.Seal(client)->id();
  }
  VINEYARD_CHECK_OK(rpc_client.DelData(nested_tuple_id, true, true));

  std::shared_ptr<InstanceStatus> status_after;
  VINEYARD_CHECK_OK(rpc_client.InstanceStatus(status_after));

  // validate memory usage
  CHECK_EQ(status_before->memory_limit, status_after->memory_limit);
  CHECK_EQ(status_before->memory_usage, status_after->memory_usage);

  LOG(INFO) << "Passed delete with rpc tests...";

  client.Disconnect();

  return 0;
}
