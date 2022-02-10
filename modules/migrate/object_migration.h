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

#ifndef MODULES_MIGRATE_OBJECT_MIGRATION_H_
#define MODULES_MIGRATE_OBJECT_MIGRATION_H_

#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "boost/asio.hpp"

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "glog/logging.h"

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/object_meta.h"
#include "migrate/protocols.h"

namespace vineyard {

namespace asio = boost::asio;
using boost::asio::ip::tcp;

class ObjectMigration {
 public:
  explicit ObjectMigration(std::vector<ObjectID> object_ids, Client& client)
      : object_ids_(object_ids), instance_id_(client.instance_id()) {}

  Status Migrate(std::unordered_map<InstanceID, InstanceID>& instance_map,
                 std::unordered_map<ObjectID, InstanceID>& object_map,
                 Client& client);

 private:
  Status getHostName(InstanceID instance_id, Client& client,
                     std::string& hostname);

  Status sendObjectMeta(ObjectID object_id, Client& client,
                        tcp::socket& socket);

  void getBlobList(json& meta_tree);

  std::vector<ObjectID> object_ids_;
  InstanceID instance_id_;

  std::set<ObjectID> blob_list_;
  std::set<ObjectID> object_list_;
};

class MigrationServer {
 public:
  explicit MigrationServer(
      std::unordered_map<InstanceID, InstanceID>& instance_map)
      : instance_map_(instance_map) {}

  Status Start(Client& client);

 private:
  ObjectID createObject(json& meta, Client& client, bool persist);

  std::unordered_map<InstanceID, InstanceID> instance_map_;
  std::unordered_map<ObjectID, json> object_map_;
  std::unordered_map<ObjectID, ObjectID> object_id_map_;
};

}  // namespace vineyard

#endif  // MODULES_MIGRATE_OBJECT_MIGRATION_H_
