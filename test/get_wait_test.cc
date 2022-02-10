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

#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/scalar.h"
#include "client/client.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./get_wait_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  using namespace std::literals::chrono_literals;  // NOLINT

  {
    // no wait
    std::thread th1([&ipc_socket]() {
      Client client;
      VINEYARD_CHECK_OK(client.Connect(ipc_socket));
      LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

      // generate a valid object id
      ObjectID id = GenerateObjectID();
      {
        ScalarBuilder<int32_t> scalar_builder(client);
        scalar_builder.SetValue(1234);

        auto scalar = std::dynamic_pointer_cast<Scalar<int32_t>>(
            scalar_builder.Seal(client));
        id = scalar->id();

        ObjectMeta meta;
        VINEYARD_CHECK_OK(client.GetMetaData(id, meta));
        VINEYARD_CHECK_OK(client.Persist(id));
      }

      std::this_thread::sleep_for(5s);
      VINEYARD_CHECK_OK(client.PutName(id, "xxx_get_wait_test_name1"));

      client.Disconnect();
    });
    std::thread th2([&ipc_socket]() {
      Client client;
      VINEYARD_CHECK_OK(client.Connect(ipc_socket));
      LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

      ObjectID id2 = 0;
      auto status = client.GetName("xxx_get_wait_test_name1", id2);
      CHECK(status.IsObjectNotExists());

      client.Disconnect();
    });
    th1.join();
    th2.join();
  }

  {
    // wait
    ObjectID id = GenerateObjectID();
    std::thread th1([&ipc_socket, &id]() mutable {
      Client client;
      VINEYARD_CHECK_OK(client.Connect(ipc_socket));
      LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

      // generate a valid object id
      {
        ScalarBuilder<int32_t> scalar_builder(client);
        scalar_builder.SetValue(1234);

        auto scalar = std::dynamic_pointer_cast<Scalar<int32_t>>(
            scalar_builder.Seal(client));
        id = scalar->id();

        ObjectMeta meta;
        VINEYARD_CHECK_OK(client.GetMetaData(id, meta));
        VINEYARD_CHECK_OK(client.Persist(id));
      }

      std::this_thread::sleep_for(5s);
      VINEYARD_CHECK_OK(client.PutName(id, "xxx_get_wait_test_name2"));

      client.Disconnect();
    });
    std::thread th2([&ipc_socket, &id]() mutable {
      Client client;
      VINEYARD_CHECK_OK(client.Connect(ipc_socket));
      LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

      ObjectID id2 = 0;
      VINEYARD_CHECK_OK(client.GetName("xxx_get_wait_test_name2", id2, true));
      CHECK_EQ(id, id2);

      client.Disconnect();
    });
    th1.join();
    th2.join();
  }

  LOG(INFO) << "Passed get-wait name tests...";

  return 0;
}
