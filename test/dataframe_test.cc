/** Copyright 2020 Alibaba Group Holding Limited.

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

#include "arrow/status.h"
#include "arrow/util/io_util.h"
#include "arrow/util/logging.h"
#include "glog/logging.h"

#include "basic/ds/dataframe.h"
#include "client/client.h"
#include "client/ds/object_meta.h"

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
    auto tb =
        std::make_shared<TensorBuilder<int>>(client, std::vector<int64_t>{100});
    builder.AddColumn("b", tb);
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
        std::dynamic_pointer_cast<TensorBuilder<int>>(builder.Column("b"));
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
  CHECK_EQ(columns.size(), 2);
  CHECK_EQ(columns[0], "a");
  CHECK_EQ(columns[1], "b");

  {
    auto column_a = std::dynamic_pointer_cast<Tensor<double>>(df->Column("a"));
    CHECK_EQ(column_a->shape()[0], 100);
    auto data = column_a->data();
    for (size_t i = 0; i < 100; ++i) {
      CHECK_EQ(data[i], i * i);
    }
  }

  {
    auto column_b = std::dynamic_pointer_cast<Tensor<int>>(df->Column("b"));
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
