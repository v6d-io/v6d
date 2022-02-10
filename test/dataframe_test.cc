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
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./dataframe_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  DataFrameBuilder builder(client);
  {
    auto tb = std::make_shared<TensorBuilder<double>>(
        client, std::vector<int64_t>{100});
    builder.AddColumn("a", tb);
  }
  {
    auto tb = std::make_shared<TensorBuilder<int64_t>>(
        client, std::vector<int64_t>{100});
    builder.AddColumn("b", tb);
  }
  {
    auto tb = std::make_shared<TensorBuilder<float>>(client,
                                                     std::vector<int64_t>{100});
    builder.AddColumn(1, tb);
  }
  {
    auto tb = std::make_shared<TensorBuilder<int32_t>>(
        client, std::vector<int64_t>{100});
    builder.AddColumn(2, tb);
  }

  // fill the column 'a'
  {
    auto column_a =
        std::dynamic_pointer_cast<TensorBuilder<double>>(builder.Column("a"));
    auto data = column_a->data();
    for (size_t i = 0; i < 100; ++i) {
      data[i] = i * i;
    }
  }

  // fill the column 'b'
  {
    auto column_a =
        std::dynamic_pointer_cast<TensorBuilder<int64_t>>(builder.Column("b"));
    auto data = column_a->data();
    for (size_t i = 0; i < 100; ++i) {
      data[i] = i * i * i;
    }
  }

  // fill the column 1
  {
    auto column_a =
        std::dynamic_pointer_cast<TensorBuilder<float>>(builder.Column(1));
    auto data = column_a->data();
    for (size_t i = 0; i < 100; ++i) {
      data[i] = i * i;
    }
  }

  // fill the column 2
  {
    auto column_a =
        std::dynamic_pointer_cast<TensorBuilder<int32_t>>(builder.Column(2));
    auto data = column_a->data();
    for (size_t i = 0; i < 100; ++i) {
      data[i] = i * i * i;
    }
  }

  auto seal_df = builder.Seal(client);
  VINEYARD_CHECK_OK(client.Persist(seal_df->id()));

  auto df =
      std::dynamic_pointer_cast<DataFrame>(client.GetObject(seal_df->id()));

  auto const& columns = df->Columns();
  CHECK_EQ(columns.size(), 4);
  CHECK_EQ(columns[0], "a");
  CHECK_EQ(columns[1], "b");
  CHECK_EQ(columns[2], 1);
  CHECK_EQ(columns[3], 2);

  {
    auto column_a = std::dynamic_pointer_cast<Tensor<double>>(df->Column("a"));
    CHECK_EQ(column_a->shape()[0], 100);
    auto data = column_a->data();
    for (size_t i = 0; i < 100; ++i) {
      CHECK_DOUBLE_EQ(data[i], i * i);
    }
  }

  {
    auto column_b = std::dynamic_pointer_cast<Tensor<int64_t>>(df->Column("b"));
    CHECK_EQ(column_b->shape()[0], 100);
    auto data = column_b->data();
    for (size_t i = 0; i < 100; ++i) {
      CHECK_EQ(data[i], i * i * i);
    }
  }

  {
    auto column_a = std::dynamic_pointer_cast<Tensor<float>>(df->Column(1));
    CHECK_EQ(column_a->shape()[0], 100);
    auto data = column_a->data();
    for (size_t i = 0; i < 100; ++i) {
      CHECK_DOUBLE_EQ(data[i], i * i);
    }
  }

  {
    auto column_b = std::dynamic_pointer_cast<Tensor<int32_t>>(df->Column(2));
    CHECK_EQ(column_b->shape()[0], 100);
    auto data = column_b->data();
    for (size_t i = 0; i < 100; ++i) {
      CHECK_EQ(data[i], i * i * i);
    }
  }

  LOG(INFO) << "Passed dataframe tests...";

  client.Disconnect();

  return 0;
}
