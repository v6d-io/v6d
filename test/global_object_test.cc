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

#include "basic/ds/dataframe.h"
#include "basic/ds/tensor.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

void testGlobalTensor(Client& client) {
  // make a tensor
  ObjectID tensor_id = InvalidObjectID();
  {
    TensorBuilder<double> builder(client, {2, 3});
    double* data = builder.data();
    for (int i = 0; i < 6; ++i) {
      data[i] = i;
    }
    auto sealed =
        std::dynamic_pointer_cast<Tensor<double>>(builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(sealed->id()));
    tensor_id = sealed->id();
  }

  // make a global tensor
  ObjectID global_tensor_id = InvalidObjectID();
  {
    GlobalTensorBuilder builder(client);
    builder.set_partition_shape({1, 1});
    builder.set_shape({2, 3});
    builder.AddPartition(tensor_id);
    global_tensor_id = builder.Seal(client)->id();
  }

  {
    std::shared_ptr<Tensor<double>> tensor;
    std::shared_ptr<GlobalTensor> global_tensor;
    tensor = client.GetObject<Tensor<double>>(tensor_id);
    global_tensor = client.GetObject<GlobalTensor>(global_tensor_id);

    CHECK(!tensor->meta().IsGlobal());
    CHECK(global_tensor->meta().IsGlobal());
  }
}

void testGlobalDataFrame(Client& client) {
  // make a dataframe
  ObjectID dataframe_id = InvalidObjectID();
  {
    DataFrameBuilder builder(client);
    {
      auto tb = std::make_shared<TensorBuilder<double>>(
          client, std::vector<int64_t>{100});
      builder.AddColumn("a", tb);
    }
    auto sealed = std::dynamic_pointer_cast<DataFrame>(builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(sealed->id()));
    dataframe_id = sealed->id();
  }

  // make a global dataframe
  ObjectID global_dataframe_id = InvalidObjectID();
  {
    GlobalDataFrameBuilder builder(client);
    builder.set_partition_shape(1, 1);
    builder.AddPartition(dataframe_id);
    global_dataframe_id = builder.Seal(client)->id();
  }

  {
    std::shared_ptr<DataFrame> dataframe;
    std::shared_ptr<GlobalDataFrame> global_dataframe;
    dataframe = client.GetObject<DataFrame>(dataframe_id);
    global_dataframe = client.GetObject<GlobalDataFrame>(global_dataframe_id);

    CHECK(!dataframe->meta().IsGlobal());
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
    auto sealed = std::dynamic_pointer_cast<DataFrame>(builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(sealed->id()));
    dataframe_id = sealed->id();
  }

  // make a global dataframe
  {
    GlobalDataFrameBuilder builder(client);
    builder.set_partition_shape(1, 1);
    builder.AddPartition(dataframe_id);
    global_dataframe_id = builder.Seal(client)->id();
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
    auto sealed = std::dynamic_pointer_cast<DataFrame>(builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(sealed->id()));
    dataframe_id = sealed->id();
  }

  // make a global dataframe
  {
    GlobalDataFrameBuilder builder(client);
    builder.set_partition_shape(1, 1);
    builder.AddPartition(dataframe_id);
    global_dataframe_id = builder.Seal(client)->id();
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
