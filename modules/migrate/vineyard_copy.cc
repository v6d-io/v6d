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

#include <signal.h>

#include <iostream>

#include "boost/algorithm/string.hpp"

#include "client/client.h"
#include "common/util/callback.h"
#include "common/util/flags.h"
#include "common/util/logging.h"
#include "migrate/flags.h"
#include "migrate/object_migration.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char* argv[]) {
  sigset(SIGINT, SIG_DFL);
  FLAGS_stderrthreshold = 0;
  flags::SetUsageMessage("Usage: vineyard_copy [options]");
  flags::ParseCommandLineFlags(&argc, &argv, true);
  logging::InitGoogleLogging("vineyard_copy");
  std::vector<std::string> w_list;
  std::vector<ObjectID> object_id_list;
  boost::split(w_list, FLAGS_object_list, boost::is_any_of(" ,\t"));
  for (auto& str_id : w_list) {
    object_id_list.emplace_back(ObjectIDFromString(str_id));
  }
  Client client;
  VINEYARD_CHECK_OK(client.Connect(FLAGS_ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << FLAGS_ipc_socket;
  InstanceID instance_id = client.instance_id();

  json instance_map_tree = json::parse(FLAGS_instance_map);
  std::unordered_map<InstanceID, InstanceID> instance_map;
  for (auto const& instance : instance_map_tree.items()) {
    InstanceID src_instance = ObjectIDFromString(instance.key());
    InstanceID dst_instance =
        instance_map_tree[instance.key()].get<InstanceID>();
    instance_map.emplace(src_instance, dst_instance);
  }

  std::unordered_map<ObjectID, ObjectID> local_object_map;

  for (auto it = instance_map.begin(); it != instance_map.end(); it++) {
    if (it->first == instance_id) {
      ObjectMigration migrate_client(object_id_list, client);
      VINEYARD_CHECK_OK(
          migrate_client.Migrate(instance_map, local_object_map, client));
      break;
    } else if (it->second == instance_id) {
      MigrationServer migrate_server(instance_map);
      VINEYARD_CHECK_OK(migrate_server.Start(client));
      break;
    }
  }
  return 0;
}
