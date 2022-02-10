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

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./server_status_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  std::shared_ptr<InstanceStatus> instance_status;
  VINEYARD_CHECK_OK(client.InstanceStatus(instance_status));
  CHECK_GT(instance_status->memory_limit, 0);
  CHECK_GT(instance_status->memory_limit, instance_status->memory_usage);
  CHECK_EQ(instance_status->instance_id, client.instance_id());

  std::vector<InstanceID> instances;
  VINEYARD_CHECK_OK(client.Instances(instances));
  CHECK_GT(instances.size(), 0);
  CHECK(std::find(instances.begin(), instances.end(), client.instance_id()) !=
        instances.end());

  std::map<InstanceID, json> cluster;
  VINEYARD_CHECK_OK(client.ClusterInfo(cluster));
  CHECK(!cluster.empty());
  CHECK(!cluster[client.instance_id()].empty());

  LOG(INFO) << "Passed server status tests...";

  client.Disconnect();

  return 0;
}
