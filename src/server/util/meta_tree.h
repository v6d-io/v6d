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

#ifndef SRC_SERVER_UTIL_META_TREE_H_
#define SRC_SERVER_UTIL_META_TREE_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "server/services/meta_service.h"

namespace vineyard {

namespace meta_tree {

enum class NodeType {
  Value = 0,
  Link = 1,
  InvalidType = 15,
};

Status GetData(const json& tree, const std::string& instance_name,
               const ObjectID id, json& sub_tree,
               InstanceID const& current_instance_id = UnspecifiedInstanceID());
Status GetData(const json& tree, const std::string& instance_name,
               const std::string& name, json& sub_tree,
               InstanceID const& current_instance_id = UnspecifiedInstanceID());
Status ListData(const json& tree, const std::string& instance_name,
                const std::string& pattern, bool const regex,
                size_t const limit, json& tree_group);
Status ListAllData(const json& tree, std::vector<ObjectID>& objects);
Status IfPersist(const json& tree, const ObjectID id, bool& persist);
Status Exists(const json& tree, const ObjectID id, bool& exists);

Status PutDataOps(const json& tree, const std::string& instance_name,
                  const ObjectID id, const json& sub_tree,
                  std::vector<IMetaService::op_t>& ops,
                  InstanceID& computed_instance_id);

Status PersistOps(const json& tree, const std::string& instance_name,
                  const ObjectID id, std::vector<IMetaService::op_t>& ops);

Status DelDataOps(const json& tree, const ObjectID id,
                  std::vector<IMetaService::op_t>& ops, bool& sync_remote);

Status DelDataOps(const json& tree, const std::set<ObjectID>& ids,
                  std::vector<IMetaService::op_t>& ops, bool& sync_remote);

Status DelDataOps(const json& tree, const std::vector<ObjectID>& ids,
                  std::vector<IMetaService::op_t>& ops, bool& sync_remote);

Status DelDataOps(const json& tree, const std::string& name,
                  std::vector<IMetaService::op_t>& ops, bool& sync_remote);

Status ShallowCopyOps(const json& tree, const ObjectID id,
                      const json& extra_metadata, const ObjectID target,
                      std::vector<IMetaService::op_t>& ops, bool& transient);

Status FilterAtInstance(const json& tree, const InstanceID& instance_id,
                        std::vector<ObjectID>& objects);

Status DecodeObjectID(const json& tree, const std::string& instance_name,
                      const std::string& value, ObjectID& object_id);

bool HasEquivalent(const json& tree, ObjectID const object_id,
                   ObjectID& equivalent);

bool MatchTypeName(bool regex, std::string const& pattern,
                   std::string const& type);

std::string EncodeValue(std::string const&);

}  // namespace meta_tree

}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_META_TREE_H_
