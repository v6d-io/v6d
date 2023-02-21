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

#include "basic/ds/dataframe.h"
#include "basic/ds/tensor.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

void testGlobalTensor(Client& client) {
  // make a tensor
  ObjectID tensor_id1 = InvalidObjectID(), tensor_id2 = InvalidObjectID();
  {
    TensorBuilder<double> builder(client, {2, 3});
    double* data = builder.data();
    for (int i = 0; i < 6; ++i) {
      data[i] = i;
    }
    std::shared_ptr<Object> sealed;
    VINEYARD_CHECK_OK(builder.Seal(client, sealed));
    tensor_id1 = sealed->id();
  }

  {
    TensorBuilder<double> builder(client, {2, 3});
    double* data = builder.data();
    for (int i = 0; i < 6; ++i) {
      data[i] = i;
    }
    std::shared_ptr<Object> sealed;
    VINEYARD_CHECK_OK(builder.Seal(client, sealed));
    tensor_id2 = sealed->id();
  }

  // make a global tensor
  ObjectID global_tensor_id = InvalidObjectID();
  {
    GlobalTensorBuilder builder(client);
    builder.set_partition_shape({1, 2});
    builder.set_shape({2, 3});
    builder.AddMember(tensor_id1);
    builder.AddMember(tensor_id2);
    builder.SetGlobal(true);
    std::shared_ptr<Object> sealed;
    VINEYARD_CHECK_OK(builder.Seal(client, sealed));
    global_tensor_id = sealed->id();
  }

  {
    std::shared_ptr<GlobalTensor> global_tensor;
    VINEYARD_CHECK_OK(client.GetObject(global_tensor_id, global_tensor));
    // global_tensor = client.GetObject<GlobalTensor>(global_tensor_id);

    std::vector<ObjectID> chunks{tensor_id1, tensor_id2};

    size_t local_index = 0;
    for (auto iter = global_tensor->LocalBegin();
         iter != global_tensor->LocalEnd(); iter.NextLocal()) {
      CHECK_EQ(chunks[local_index++], iter->id());
      CHECK(iter->meta().IsLocal());
      CHECK(!iter->meta().IsGlobal());
    }

    CHECK(global_tensor->meta().IsGlobal());
  }
}

void testGlobalDataFrame(Client& client) {
  // make a dataframe
  ObjectID dataframe_id1 = InvalidObjectID(), dataframe_id2 = InvalidObjectID();
  {
    DataFrameBuilder builder(client);
    {
      auto tb = std::make_shared<TensorBuilder<double>>(
          client, std::vector<int64_t>{100});
      builder.AddColumn("a", tb);
    }
    std::shared_ptr<Object> sealed;
    VINEYARD_CHECK_OK(builder.Seal(client, sealed));
    VINEYARD_CHECK_OK(client.Persist(sealed->id()));
    dataframe_id1 = sealed->id();
  }

  {
    DataFrameBuilder builder(client);
    {
      auto tb = std::make_shared<TensorBuilder<double>>(
          client, std::vector<int64_t>{100});
      builder.AddColumn("a", tb);
    }
    std::shared_ptr<Object> sealed;
    VINEYARD_CHECK_OK(builder.Seal(client, sealed));
    VINEYARD_CHECK_OK(client.Persist(sealed->id()));
    dataframe_id2 = sealed->id();
  }

  // make a global dataframe
  ObjectID global_dataframe_id = InvalidObjectID();
  {
    GlobalDataFrameBuilder builder(client);
    builder.set_partition_shape(2, 1);
    builder.AddMember(dataframe_id1);
    builder.AddMember(dataframe_id2);
    builder.SetGlobal(true);
    std::shared_ptr<Object> sealed;
    VINEYARD_CHECK_OK(builder.Seal(client, sealed));
    global_dataframe_id = sealed->id();
  }

  {
    std::shared_ptr<GlobalDataFrame> global_dataframe;
    global_dataframe = client.GetObject<GlobalDataFrame>(global_dataframe_id);

    std::vector<ObjectID> chunks{dataframe_id1, dataframe_id2};
    size_t chunk_index = 0;
    for (auto iter = global_dataframe->LocalBegin();
         iter != global_dataframe->LocalEnd(); iter.NextLocal()) {
      CHECK_EQ(chunks[chunk_index++], iter->id());
      CHECK(iter->meta().IsLocal());
      CHECK(!iter->meta().IsGlobal());
    }

    CHECK(global_dataframe->meta().IsGlobal());
  }
}

void testDelete(Client& client) {
  ObjectID dataframe_id = InvalidObjectID();
  ObjectID global_dataframe_id = InvalidObjectID();

  // make a dataframe
  {
    DataFrameBuilder builder(client);
    {
      auto tb = std::make_shared<TensorBuilder<double>>(
          client, std::vector<int64_t>{100});
      builder.AddColumn("a", tb);
    }
    std::shared_ptr<Object> sealed;
    VINEYARD_CHECK_OK(builder.Seal(client, sealed));
    VINEYARD_CHECK_OK(client.Persist(sealed->id()));
    dataframe_id = sealed->id();
  }

  // make a global dataframe
  {
    GlobalDataFrameBuilder builder(client);
    builder.set_partition_shape(1, 1);
    builder.AddMember(dataframe_id);
    builder.SetGlobal(true);
    std::shared_ptr<Object> sealed;
    VINEYARD_CHECK_OK(builder.Seal(client, sealed));
    global_dataframe_id = sealed->id();
  }

  {
    bool exists = false;
    VINEYARD_CHECK_OK(client.Exists(dataframe_id, exists));
    CHECK(exists);
    VINEYARD_CHECK_OK(client.Exists(global_dataframe_id, exists));
    CHECK(exists);

    LOG(INFO) << "delete global dataframe id: " << global_dataframe_id << ": "
              << ObjectIDToString(global_dataframe_id);
    VINEYARD_CHECK_OK(client.DelData(global_dataframe_id, false, true));
    VINEYARD_CHECK_OK(client.Exists(global_dataframe_id, exists));
    CHECK(!exists);
    VINEYARD_CHECK_OK(client.Exists(dataframe_id, exists));
    CHECK(!exists);
  }

  // make a dataframe
  {
    DataFrameBuilder builder(client);
    {
      auto tb = std::make_shared<TensorBuilder<double>>(
          client, std::vector<int64_t>{100});
      builder.AddColumn("a", tb);
    }
    std::shared_ptr<Object> sealed;
    VINEYARD_CHECK_OK(builder.Seal(client, sealed));
    VINEYARD_CHECK_OK(client.Persist(sealed->id()));
    dataframe_id = sealed->id();
  }

  // make a global dataframe
  {
    GlobalDataFrameBuilder builder(client);
    builder.set_partition_shape(1, 1);
    builder.AddMember(dataframe_id);
    builder.SetGlobal(true);
    std::shared_ptr<Object> sealed;
    VINEYARD_CHECK_OK(builder.Seal(client, sealed));
    global_dataframe_id = sealed->id();
  }

  {
    bool exists = false;
    VINEYARD_CHECK_OK(client.Exists(dataframe_id, exists));
    CHECK(exists);
    VINEYARD_CHECK_OK(client.Exists(global_dataframe_id, exists));
    CHECK(exists);

    LOG(INFO) << "delete dataframe chunk id: " << dataframe_id << ": "
              << ObjectIDToString(dataframe_id);
    VINEYARD_CHECK_OK(client.DelData(dataframe_id, false, true));
    VINEYARD_CHECK_OK(client.Exists(global_dataframe_id, exists));
    CHECK(exists);
    VINEYARD_CHECK_OK(client.Exists(dataframe_id, exists));
    CHECK(exists);

    LOG(INFO) << "delete dataframe chunk id with deep: " << dataframe_id << ": "
              << ObjectIDToString(dataframe_id);
    VINEYARD_CHECK_OK(client.DelData(dataframe_id, true, true));
    VINEYARD_CHECK_OK(client.Exists(global_dataframe_id, exists));
    CHECK(!exists);
    VINEYARD_CHECK_OK(client.Exists(dataframe_id, exists));
    CHECK(!exists);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./global_object_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  testGlobalTensor(client);
  testGlobalDataFrame(client);
  testDelete(client);

  client.Disconnect();

  LOG(INFO) << "Passed global object tests...";

  return 0;
}
