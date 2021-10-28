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

#include "arrow/status.h"
#include "arrow/stl.h"
#include "arrow/util/config.h"
#include "arrow/util/io_util.h"
#include "arrow/util/logging.h"

#include "basic/ds/arrow-v2.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./arrow_data_structure_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  {
    LOG(INFO) << "#########  Double Test #############";
    arrow::DoubleBuilder b1;
    CHECK_ARROW_ERROR(b1.AppendValues({1.5, 2.5, 3.5, 4.5}));
    CHECK_ARROW_ERROR(b1.AppendNull());
    CHECK_ARROW_ERROR(b1.AppendValues({5.5}));
    std::shared_ptr<arrow::DoubleArray> a1;
    CHECK_ARROW_ERROR(b1.Finish(&a1));
    v2::NumericArrayBuilder<double> array_builder(client, a1);
    auto r1 = std::dynamic_pointer_cast<v2::NumericArray<double>>(
        array_builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(r1->id()));
    ObjectID id = r1->id();

    auto r2 = std::dynamic_pointer_cast<v2::NumericArray<double>>(
        client.GetObject(id));
    // auto internal_array =
    // std::static_pointer_cast<ConvertToArrowType<double>::ArrayType>(r2);
    auto internal_array = r2;
    // LOG(INFO) << a1->ToString();
    // LOG(INFO) << internal_array->ToString();
    CHECK(internal_array->Equals(*a1));

    LOG(INFO) << "Passed double array wrapper tests...";
  }

  client.Disconnect();

  return 0;
}
