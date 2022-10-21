/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/util/json.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

namespace meta_tree {

struct kv_t {
  std::string key;
  std::string value;
  unsigned rev;
};

struct op_t {
  enum op_type_t : unsigned { kPut = 0, kDel = 1 } op;
  kv_t kv;
  std::string ToString() const {
    std::stringstream ss;
    ss.str("");
    ss.clear();
    ss << ((op == kPut) ? "put " : "del ");
    ss << "[" << kv.rev << "] " << kv.key << " -> " << kv.value;
    return ss.str();
  }

  static op_t Del(std::string const& key) {
    return op_t{.op = op_type_t::kDel,
                .kv = kv_t{.key = key, .value = "", .rev = 0}};
  }
  static op_t Del(std::string const& key, unsigned const rev) {
    return op_t{.op = op_type_t::kDel,
                .kv = kv_t{.key = key, .value = "", .rev = rev}};
  }
  // send to etcd
  template <typename T>
  static op_t Put(std::string const& key, T const& value) {
    return op_t{
        .op = op_type_t::kPut,
        .kv = kv_t{.key = key, .value = json_to_string(json(value)), .rev = 0}};
  }
  template <typename T>
  static op_t Put(std::string const& key, json const& value) {
    return op_t{
        .op = op_type_t::kPut,
        .kv = kv_t{.key = key, .value = json_to_string(value), .rev = 0}};
  }
  // receive from etcd
  static op_t Put(std::string const& key, std::string const& value,
                  unsigned const rev) {
    return op_t{.op = op_type_t::kPut,
                .kv = kv_t{.key = key, .value = value, .rev = rev}};
  }
};

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
Status ListName(const json& tree, std::string const& pattern, bool const regex,
                size_t const limit, std::map<std::string, ObjectID>& names);
Status IfPersist(const json& tree, const ObjectID id, bool& persist);
Status Exists(const json& tree, const ObjectID id, bool& exists);

Status PutDataOps(const json& tree, const std::string& instance_name,
                  const ObjectID id, const json& sub_tree,
                  std::vector<op_t>& ops, InstanceID& computed_instance_id);

Status PersistOps(const json& tree, const std::string& instance_name,
                  const ObjectID id, std::vector<op_t>& ops);

Status DelDataOps(const json& tree, const ObjectID id, std::vector<op_t>& ops,
                  bool& sync_remote);

Status DelDataOps(const json& tree, const std::set<ObjectID>& ids,
                  std::vector<op_t>& ops, bool& sync_remote);

Status DelDataOps(const json& tree, const std::vector<ObjectID>& ids,
                  std::vector<op_t>& ops, bool& sync_remote);

Status DelDataOps(const json& tree, const std::string& name,
                  std::vector<op_t>& ops, bool& sync_remote);

Status ShallowCopyOps(const json& tree, const ObjectID id,
                      const json& extra_metadata, const ObjectID target,
                      std::vector<op_t>& ops, bool& transient);

Status FilterAtInstance(const json& tree, const InstanceID& instance_id,
                        std::vector<ObjectID>& objects);

Status DecodeObjectID(const json& tree, const std::string& instance_name,
                      const std::string& value, ObjectID& object_id);

bool HasEquivalent(const json& tree, ObjectID const object_id,
                   ObjectID& equivalent);

bool HasEquivalentWithSignature(const json& tree, Signature const signature,
                                ObjectID const object_id, ObjectID& equivalent);

bool MatchTypeName(bool regex, std::string const& pattern,
                   std::string const& type);

std::string EncodeValue(std::string const&);

}  // namespace meta_tree

}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_META_TREE_H_
