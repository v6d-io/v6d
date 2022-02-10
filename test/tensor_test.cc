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

#include "basic/ds/tensor.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./tensor_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  TensorBuilder<double> builder(client, {2, 3});
  double* data = builder.data();
  for (int i = 0; i < 6; ++i) {
    data[i] = i;
  }
  auto sealed = std::dynamic_pointer_cast<Tensor<double>>(builder.Seal(client));
  VINEYARD_CHECK_OK(client.Persist(sealed->id()));

  auto tensor = sealed->ArrowTensor();
  CHECK_EQ(tensor->shape().size(), 2);
  CHECK_EQ(tensor->shape()[0], 2);
  CHECK_EQ(tensor->shape()[1], 3);
  const double* sealed_data = sealed->data();
  for (int i = 0; i < 6; ++i) {
    CHECK_EQ(sealed_data[i], i);
  }

  LOG(INFO) << "Passed tensor tests...";

  client.Disconnect();

  return 0;
}
