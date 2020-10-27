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

#ifndef SRC_SERVER_UTIL_META_TREE_H_
#define SRC_SERVER_UTIL_META_TREE_H_

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

Status GetData(const ptree& tree, const ObjectID id, ptree& sub_tree);
Status GetData(const ptree& tree, const std::string& name, ptree& sub_tree);
Status ListData(const ptree& tree, const std::string& pattern, bool const regex,
                size_t const limit, ptree& tree_group);
Status DelData(ptree& tree, const ObjectID id);
Status DelData(ptree& tree, const std::vector<ObjectID>& ids);
Status IfPersist(const ptree& tree, const ObjectID id, bool& persist);
Status Exists(const ptree& tree, const ObjectID id, bool& exists);

Status PutDataOps(const ptree& tree, const ObjectID id, const ptree& sub_tree,
                  std::vector<IMetaService::op_t>& ops,
                  InstanceID& computed_instance_id);

Status PersistOps(const ptree& tree, const ObjectID id,
                  std::vector<IMetaService::op_t>& ops);

Status DelDataOps(const ptree& tree, const ObjectID id,
                  std::vector<IMetaService::op_t>& ops);

Status DelDataOps(const ptree& tree, const std::set<ObjectID>& ids,
                  std::vector<IMetaService::op_t>& ops);

Status DelDataOps(const ptree& tree, const std::string& name,
                  std::vector<IMetaService::op_t>& ops);

Status ShallowCopyOps(const ptree& tree, const ObjectID id,
                      const ObjectID target,
                      std::vector<IMetaService::op_t>& ops, bool& transient);

Status FilterAtInstance(const ptree& tree, const InstanceID& instance_id,
                        std::vector<ObjectID>& objects);

Status DecodeObjectID(const std::string& value, ObjectID& object_id);

}  // namespace meta_tree

}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_META_TREE_H_
