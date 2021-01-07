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

#include "server/util/meta_tree.h"

#include <fnmatch.h>

#include <regex>
#include <set>
#include <string>
#include <vector>

#include "boost/lexical_cast.hpp"

namespace boost {
// Makes the behaviour of lexical_cast compatibile with boost::property_tree.
template <>
bool lexical_cast<bool, std::string>(const std::string& arg) {
  std::istringstream ss(arg);
  bool b;
  if (std::isalpha(arg[0])) {
    ss >> std::boolalpha >> b;
  } else {
    ss >> b;
  }
  return b;
}
}  // namespace boost

namespace vineyard {

namespace meta_tree {

static void decode_value(const std::string& str, NodeType& type,
                         std::string& value) {
  if (str[0] == 'v') {
    type = NodeType::Value;
    value = str.substr(1);
  } else if (str[0] == 'l') {
    type = NodeType::Link;
    value = str.substr(1);
  } else {
    type = NodeType::InvalidType;
    value.clear();
  }
}

static void encode_value(NodeType type, const std::string& value,
                         std::string& str) {
  str.clear();
  if (type == NodeType::Value) {
    str.resize(value.size() + 1);
    str[0] = 'v';
    memcpy(&str[1], value.c_str(), value.size());
  } else if (type == NodeType::Link) {
    str.resize(value.size() + 1);
    str[0] = 'l';
    memcpy(&str[1], value.c_str(), value.size());
  }
}

static bool __attribute__((used)) is_link_node(const std::string& str) {
  return str[0] == 'l';
}

static bool __attribute__((used)) is_value_node(const std::string& str) {
  return str[0] == 'v';
}

static Status parse_link(const std::string& str, std::string& type,
                         std::string& name) {
  std::string::size_type l1 = str.find(".");
  std::string::size_type l2 = str.rfind(".");
  if (l1 == std::string::npos || l1 != l2) {
    type.clear();
    name.clear();
    LOG(ERROR) << "meta tree link invalid: " << str;
    return Status::MetaTreeLinkInvalid();
  }
  name = str.substr(0, l1);
  type = str.substr(l1 + 1);
  if (type.empty() || name.empty()) {
    LOG(ERROR) << "meta tree link invalid: " << type << ", " << name;
    type.clear();
    name.clear();
    return Status::MetaTreeLinkInvalid();
  }
  return Status::OK();
}

static void generate_link(const std::string& type, const std::string& name,
                          std::string& link) {
  // shorten the typename, but still leave it in the link, to make the metatree
  // more verbose.
  link = name + "." + type.substr(0, type.find_first_of('<'));
}

static Status get_sub_tree(const json& tree, const std::string& prefix,
                           const std::string& name, json& sub_tree) {
  if (name.find('/') != std::string::npos) {
    LOG(ERROR) << "meta tree name invalid. " << name;
    return Status::MetaTreeNameInvalid();
  }
  std::string path = prefix;
  if (!name.empty()) {
    path += "/" + name;
  }
  auto json_path = json::json_pointer(path);
  if (tree.contains(json_path)) {
    auto const& tmp_tree = tree[json_path];
    if (tmp_tree.is_object() && !tmp_tree.empty()) {
      sub_tree = tmp_tree;
      return Status::OK();
    }
  }
  return Status::MetaTreeSubtreeNotExists();
}

static bool has_sub_tree(const json& tree, const std::string& prefix,
                         const std::string& name) {
  if (name.find('/') != std::string::npos) {
    return false;
  }
  std::string path = prefix;
  if (!name.empty()) {
    path += "/" + name;
  }
  return tree.contains(json::json_pointer(path));
}

static Status del_sub_tree(json& tree, const std::string& prefix,
                           const std::string& name) {
  if (name.find('/') != std::string::npos) {
    LOG(ERROR) << "meta tree name invalid. " << name;
    return Status::MetaTreeNameInvalid();
  }
  auto path = json::json_pointer(prefix);
  if (tree.contains(path)) {
    if (tree[path].contains(name)) {
      tree[path].erase(name);
      return Status::OK();
    } else {
      LOG(ERROR) << "meta tree name doesn't exist: " << name;
      return Status::MetaTreeNameNotExists();
    }
  } else {
    LOG(ERROR) << "meta tree subtree doesn't exist." << path;
    return Status::MetaTreeSubtreeNotExists();
  }
}

static Status get_name(const json& tree, std::string& name,
                       bool const decode = false) {
  // name: get the object id
  json::const_iterator name_iter = tree.find("id");
  if (name_iter == tree.end()) {
    return Status::MetaTreeNameNotExists();
  }
  if (name_iter->is_object()) {
    LOG(ERROR) << "meta tree id invalid. " << *name_iter;
    return Status::MetaTreeNameInvalid();
  }
  name = name_iter->get_ref<std::string const&>();
  if (decode) {
    NodeType node_type = NodeType::InvalidType;
    decode_value(name, node_type, name);
    if (node_type != NodeType::Value) {
      return Status::MetaTreeNameInvalid();
    }
  }
  return Status::OK();
}

static Status get_type(const json& tree, std::string& type,
                       bool const decode = false) {
  // type: get the typename
  json::const_iterator type_iter = tree.find("typename");
  if (type_iter == tree.end()) {
    return Status::MetaTreeNameNotExists();
  }
  if (type_iter->is_object()) {
    LOG(ERROR) << "meta tree typename invalid. " << *type_iter;
    return Status::MetaTreeTypeInvalid();
  }
  type = type_iter->get_ref<std::string const&>();
  if (decode) {
    NodeType node_type = NodeType::InvalidType;
    decode_value(type, node_type, type);
    if (node_type != NodeType::Value) {
      return Status::MetaTreeTypeInvalid();
    }
  }
  return Status::OK();
}

static Status get_type_name(const json& tree, std::string& type,
                            std::string& name, bool const decode = false) {
  RETURN_ON_ERROR(get_type(tree, type, decode));
  RETURN_ON_ERROR(get_name(tree, name, decode));
  return Status::OK();
}

/**
 * In `MetaData::AddMember`, the parameter might be an object id. In such cases
 * the client doesn't have the full metadata json of the object, there will be
 * just an `ObjectID`.
 */
static bool is_meta_placeholder(const json& tree) {
  return tree.is_object() && tree.size() == 1 && tree.contains("id");
}

/**
 * Get metadata for an object "recursively".
 */
Status GetData(const json& tree, const ObjectID id, json& sub_tree) {
  return GetData(tree, VYObjectIDToString(id), sub_tree);
}

/**
 * Get metadata for an object "recursively".
 */
Status GetData(const json& tree, const std::string& name, json& sub_tree) {
  json tmp_tree;
  sub_tree.clear();
  Status status = get_sub_tree(tree, "/data", name, tmp_tree);
  if (!status.ok()) {
    return status;
  }
  for (auto const& item : json::iterator_wrapper(tmp_tree)) {
    if (!item.value().is_string()) {
      sub_tree[item.key()] = item.value();
      continue;
    }
    std::string const& item_value = item.value().get_ref<std::string const&>();
    NodeType type;
    std::string value;
    decode_value(item_value, type, value);
    if (type == NodeType::Value) {
      sub_tree[item.key()] = value;
    } else if (type == NodeType::Link) {
      std::string sub_sub_tree_type, sub_sub_tree_name;
      status = parse_link(value, sub_sub_tree_type, sub_sub_tree_name);
      if (!status.ok()) {
        sub_tree.clear();
        return status;
      }
      json sub_sub_tree;
      status = GetData(tree, sub_sub_tree_name, sub_sub_tree);
      if (status.ok()) {
        sub_tree[item.key()] = sub_sub_tree;
      } else {
        ObjectID sub_sub_tree_id = VYObjectIDFromString(sub_sub_tree_name);
        if (IsBlob(sub_sub_tree_id) && status.IsMetaTreeSubtreeNotExists()) {
          // make an empty blob
          sub_sub_tree["id"] = VYObjectIDToString(EmptyBlobID());
          sub_sub_tree["typename"] = "vineyard::Blob";
          sub_sub_tree["length"] = 0;
          sub_sub_tree["nbytes"] = 0;
          sub_sub_tree["instance_id"] = UnspecifiedInstanceID();
          sub_sub_tree["transient"] = true;
          sub_tree[item.key()] = sub_sub_tree;
        } else {
          sub_tree.clear();
          return status;
        }
      }
    } else {
      return Status::MetaTreeTypeInvalid();
    }
  }
  sub_tree["id"] = name;
  return Status::OK();
}

Status ListData(const json& tree, std::string const& pattern, bool const regex,
                size_t const limit, json& tree_group) {
  if (!tree.contains("data")) {
    return Status::OK();
  }

  size_t found = 0;

  std::regex regex_pattern;
  if (regex) {
    // pre-compile regex pattern, and for invalid regex pattern, return nothing.
    try {
      regex_pattern = std::regex(pattern);
    } catch (std::regex_error const&) { return Status::OK(); }
  }

  for (auto const& item : json::iterator_wrapper(tree["data"])) {
    if (found >= limit) {
      break;
    }

    if (!item.value().is_object() || item.value().empty()) {
      LOG(INFO) << "Object meta shouldn't be empty";
      return Status::MetaTreeInvalid();
    }
    std::string type;
    RETURN_ON_ERROR(get_type(item.value(), type, true));

    // match type on pattern
    bool matched = false;
    if (regex) /* regex match */ {
      std::cmatch __m;
      matched = std::regex_match(type.c_str(), __m, regex_pattern);
    } else /* wildcard match */ {
      // https://www.man7.org/linux/man-pages/man3/fnmatch.3.html
      matched = fnmatch(pattern.c_str(), type.c_str(), 0) == 0;
    }

    if (matched) {
      found += 1;
      json object_meta_tree;
      RETURN_ON_ERROR(GetData(tree, item.key(), object_meta_tree));
      tree_group[item.key()] = object_meta_tree;
    }
  }
  return Status::OK();
}

Status DelData(json& tree, const ObjectID id) {
  std::string name = VYObjectIDToString(id);
  return del_sub_tree(tree, "/data", name);
}

Status DelData(json& tree, const std::vector<ObjectID>& ids) {
  // FIXME: use a more efficient implmentation.
  for (auto const& id : ids) {
    auto s = DelData(tree, id);
    if (!s.ok()) {
      return s;
    }
  }
  return Status::OK();
}

Status DelDataOps(const json& tree, const ObjectID id,
                  std::vector<IMetaService::op_t>& ops) {
  return DelDataOps(tree, VYObjectIDToString(id), ops);
}

Status DelDataOps(const json& tree, const std::set<ObjectID>& ids,
                  std::vector<IMetaService::op_t>& ops) {
  // FIXME: use a more efficient implmentation.
  for (auto const& id : ids) {
    auto s = DelDataOps(tree, id, ops);
    if (!s.ok()) {
      return s;
    }
  }
  return Status::OK();
}

Status DelDataOps(const json& tree, const std::string& name,
                  std::vector<IMetaService::op_t>& ops) {
  std::string data_prefix = "/data";
  auto json_path = json::json_pointer(data_prefix);
  if (tree.contains(json_path)) {
    auto const& data_tree = tree[json_path];
    if (data_tree.contains(name)) {
      // erase from etcd
      ops.emplace_back(IMetaService::op_t::Del(data_prefix + "/" + name));
      return Status::OK();
    }
  }
  return Status::MetaTreeSubtreeNotExists();
}

static void generate_put_ops(const json& meta, const json& diff,
                             const std::string& name,
                             std::vector<IMetaService::op_t>& ops) {
  std::string key_prefix = "/data" + std::string("/") + name + "/";
  for (auto const& item : json::iterator_wrapper(diff)) {
    if (item.value().is_object() && !item.value().empty()) {
      std::string sub_type, sub_name;
      VINEYARD_SUPPRESS(get_type_name(item.value(), sub_type, sub_name));
      if (!has_sub_tree(meta, "/data", sub_name)) {
        generate_put_ops(meta, item.value(), sub_name, ops);
      }
      std::string link;
      generate_link(sub_type, sub_name, link);
      std::string encoded_value;
      encode_value(NodeType::Link, link, encoded_value);
      ops.emplace_back(
          IMetaService::op_t::Put(key_prefix + item.key(), encoded_value));
    } else {
      // don't repeat "id" in the etcd kvs.
      if (item.key() == "id") {
        continue;
      }
      std::string key = key_prefix + item.key();
      if (item.value().is_string()) {
        std::string encoded_value;
        encode_value(NodeType::Value,
                     item.value().get_ref<std::string const&>(), encoded_value);
        ops.emplace_back(IMetaService::op_t::Put(key, encoded_value));
      } else {
        ops.emplace_back(IMetaService::op_t::Put(key, item.value()));
      }
    }
  }
}

static void generate_persist_ops(json& diff, const std::string& name,
                                 std::vector<IMetaService::op_t>& ops,
                                 std::set<std::string>& dedup) {
  std::string data_key = "/data" + std::string("/") + name;
  if (dedup.find(data_key) != dedup.end()) {
    return;
  }
  for (auto& item : json::iterator_wrapper(diff)) {
    if (item.value().is_object() &&
        !item.value().empty()) /* build link, and recursively generate */ {
      std::string sub_type, sub_name;
      VINEYARD_SUPPRESS(get_type_name(item.value(), sub_type, sub_name));
      // Don't persist blob into etcd, but the link cannot be omitted.
      if (item.value()["transient"].get<bool>() &&
          sub_type != "vineyard::Blob") {
        // otherwise, skip recursively generate ops
        generate_persist_ops(item.value(), sub_name, ops, dedup);
      }
      std::string link;
      generate_link(sub_type, sub_name, link);
      std::string encoded_value;
      encode_value(NodeType::Link, link, encoded_value);
      diff[item.key()] = encoded_value;
    } else /* do value transformation (encoding) */ {
      json value_to_persist;
      if (item.key() == "transient") {
        diff[item.key()] = false;
      } else if (item.value().is_string()) {
        std::string encoded_value;
        encode_value(NodeType::Value,
                     item.value().get_ref<std::string const&>(), encoded_value);
        diff[item.key()] = encoded_value;
      }
    }
  }
  // don't repeat "id" in the etcd kvs.
  diff.erase("id");
  ops.emplace_back(IMetaService::op_t::Put(data_key, diff));
  dedup.emplace(data_key);
}

/**
 * Returns:
 *
 *  diff: diff json
 *  instance_id: instance_id of members and the object itself, can represents
 *               the final instance_id of the object.
 */
static Status diff_data_meta_tree(const json& meta,
                                  const std::string& sub_tree_name,
                                  const json& sub_tree, json& diff,
                                  InstanceID& instance_id) {
  json old_sub_tree;
  Status status = get_sub_tree(meta, "/data", sub_tree_name, old_sub_tree);

  if (!status.ok()) {
    if (status.IsMetaTreeSubtreeNotExists()) {
      diff["transient"] = true;
    } else {
      return status;
    }
  }

  // when put data using meta placeholder, the object it points to must exist,
  // and no need to perform diff.
  if (is_meta_placeholder(sub_tree)) {
    if (status.ok()) {
      std::string sub_tree_type;
      RETURN_ON_ERROR(get_type(old_sub_tree, sub_tree_type, true));
      diff["id"] = sub_tree_name;
      diff["typename"] = sub_tree_type;
      instance_id = old_sub_tree["instance_id"].get<InstanceID>();
    }
    return status;
  }

  // recompute instance_id: check if it refers remote objects.
  instance_id = sub_tree["instance_id"].get<InstanceID>();

  for (auto const& item : json::iterator_wrapper(sub_tree)) {
    // don't diff on id, typename and instance_id and don't update them, that
    // means, we can only update member's meta, cannot update the whole member
    // itself.
    if (item.key() == "id" || item.key() == "typename" ||
        item.key() == "instance_id") {
      continue;
    }

    if (!item.value().is_object() /* plain value */) {
      const json& new_value = item.value();
      if (status.ok() /* old meta exists */) {
        if (old_sub_tree.contains(item.key())) {
          auto old_value = old_sub_tree[item.key()];

          if (old_value.is_string()) {
            NodeType old_value_type;
            std::string old_value_decoded;
            decode_value(old_value.get_ref<std::string const&>(),
                         old_value_type, old_value_decoded);
            old_value = json(old_value_decoded);
          }

          bool require_update = false;
          if (item.key() == "transient") {
            // put_data wan't make persist value becomes transient, since the
            // info in the client may be out-of-date.
            require_update = old_value == true && old_value != new_value;
          } else {
            require_update = old_value != new_value;
          }

          if (require_update) {
            VLOG(10) << "DIFF: " << item.key() << ": " << old_value << " -> "
                     << new_value;
            diff[item.key()] = new_value;
          }
        } else {
          VLOG(10) << "DIFF: " << item.key() << ": [none] -> " << new_value;
          diff[item.key()] = new_value;
        }
      } else if (status.IsMetaTreeSubtreeNotExists()) {
        VLOG(10) << "DIFF: " << item.key() << ": [none] -> " << new_value;
        diff[item.key()] = new_value;
      } else {
        return status;
      }
    } else /* member object */ {
      const json& sub_sub_tree = item.value();

      // original corresponding field must be a member not a key-value
      auto mb_old_sub_sub_tree = old_sub_tree.find(item.key());
      if (status.ok() /* old meta exists */) {
        if (mb_old_sub_sub_tree != old_sub_tree.end() &&
            !mb_old_sub_sub_tree->is_string() &&
            !is_link_node(mb_old_sub_sub_tree->get_ref<std::string const&>())) {
          return Status::MetaTreeInvalid();
        }
      }

      std::string sub_sub_tree_name;
      RETURN_ON_ERROR(get_name(sub_sub_tree, sub_sub_tree_name));

      json diff_sub_tree;
      InstanceID sub_instance_id;
      RETURN_ON_ERROR(diff_data_meta_tree(meta, sub_sub_tree_name, sub_sub_tree,
                                          diff_sub_tree, sub_instance_id));

      if (instance_id != sub_instance_id) {
        instance_id = UnspecifiedInstanceID();
      }

      if (status.ok() /* old meta exists */) {
        if (diff_sub_tree.is_object() && !diff_sub_tree.empty()) {
          diff[item.key()] = diff_sub_tree;
        }
      } else if (status.IsMetaTreeSubtreeNotExists()) {
        if (!is_meta_placeholder(sub_sub_tree)) {
          // if sub_sub_tree is placeholder, the diff already contains id and
          // typename
          std::string sub_sub_tree_type;
          RETURN_ON_ERROR(get_type_name(sub_sub_tree, sub_sub_tree_type,
                                        sub_sub_tree_name));
          diff_sub_tree["id"] = sub_sub_tree_name;
          diff_sub_tree["typename"] = sub_sub_tree_type;
        }
        diff[item.key()] = diff_sub_tree;
      } else {
        return status;
      }
    }
  }
  if (diff.is_object() && !diff.empty() &&
      status.IsMetaTreeSubtreeNotExists()) {
    // must not be meta placeholder
    std::string sub_tree_type;
    RETURN_ON_ERROR(get_type(sub_tree, sub_tree_type));

    diff["id"] = sub_tree_name;
    diff["typename"] = sub_tree_type;
    diff["instance_id"] = instance_id;
  }
  return Status::OK();
}

static void persist_meta_tree(const json& sub_tree, json& diff) {
  // NB: we don't need to track which objects are persist since the json
  // cached in the server will be updated by the background watcher task.
  if (sub_tree["transient"].get<bool>()) {
    for (auto const& item : json::iterator_wrapper(sub_tree)) {
      if (item.value().is_object() && !item.value().empty()) {
        const json& sub_sub_tree = item.value();
        // recursive
        json sub_diff;
        persist_meta_tree(sub_sub_tree, sub_diff);
        if (sub_diff.is_object() && !sub_diff.empty()) {
          diff[item.key()] = sub_diff;
        } else {
          // will be used to generate the link.
          diff[item.key()] = sub_sub_tree;
        }
      } else {
        diff[item.key()] = item.value();
      }
    }
  }
}

Status PutDataOps(const json& tree, const ObjectID id, const json& sub_tree,
                  std::vector<IMetaService::op_t>& ops,
                  InstanceID& computed_instance_id) {
  json diff;
  std::string name = VYObjectIDToString(id);
  // recompute instance_id: check if it refers remote objects.
  Status status =
      diff_data_meta_tree(tree, name, sub_tree, diff, computed_instance_id);

  if (!status.ok()) {
    return status;
  }

  if (diff.is_null() || (diff.is_object() && diff.empty())) {
    return Status::OK();
  }

  generate_put_ops(tree, diff, name, ops);
  return Status::OK();
}

Status PersistOps(const json& tree, const ObjectID id,
                  std::vector<IMetaService::op_t>& ops) {
  json sub_tree, diff;
  Status status = GetData(tree, id, sub_tree);
  if (!status.ok()) {
    return status;
  }
  persist_meta_tree(sub_tree, diff);

  if (diff.is_null() || (diff.is_object() && diff.empty())) {
    return Status::OK();
  }

  std::string name = VYObjectIDToString(id);
  std::set<std::string> dedup;
  generate_persist_ops(diff, name, ops, dedup);
  return Status::OK();
}

Status Exists(const json& tree, const ObjectID id, bool& exists) {
  std::string name = VYObjectIDToString(id);
  exists = has_sub_tree(tree, "/data", name);
  return Status::OK();
}

Status ShallowCopyOps(const json& tree, const ObjectID id,
                      const ObjectID target,
                      std::vector<IMetaService::op_t>& ops, bool& transient) {
  std::string name = VYObjectIDToString(id);
  json tmp_tree;
  RETURN_ON_ERROR(get_sub_tree(tree, "/data", name, tmp_tree));
  RETURN_ON_ASSERT(
      tmp_tree.contains("transient") && tmp_tree["transient"].is_boolean(),
      "The 'transient' should a plain bool value");
  transient = tmp_tree["transient"].get<bool>();
  std::string key_prefix =
      "/data" + std::string("/") + VYObjectIDToString(target) + "/";
  for (auto const& item : json::iterator_wrapper(tmp_tree)) {
    ops.emplace_back(
        IMetaService::op_t::Put(key_prefix + item.key(), item.value()));
  }
  return Status::OK();
}

Status IfPersist(const json& tree, const ObjectID id, bool& persist) {
  std::string name = VYObjectIDToString(id);
  json tmp_tree;
  Status status = get_sub_tree(tree, "/data", name, tmp_tree);
  if (status.ok()) {
    RETURN_ON_ASSERT(
        tmp_tree.contains("transient") && tmp_tree["transient"].is_boolean(),
        "The 'transient' should a plain boolean value");
    persist = !tmp_tree["transient"].get<bool>();
  }
  return status;
}

Status FilterAtInstance(const json& tree, const InstanceID& instance_id,
                        std::vector<ObjectID>& objects) {
  if (tree.contains("data")) {
    for (auto const& item : json::iterator_wrapper(tree["data"])) {
      if (item.value().is_object() && !item.value().empty()) {
        if (item.value().contains("instance_id") &&
            item.value()["instance_id"].get<InstanceID>() == instance_id) {
          objects.emplace_back(VYObjectIDFromString(item.key()));
        }
      }
    }
  }
  return Status::OK();
}

Status DecodeObjectID(const std::string& value, ObjectID& object_id) {
  meta_tree::NodeType type;
  std::string link_value;
  decode_value(value, type, link_value);
  if (type == NodeType::Link) {
    std::string type_of_value, name_of_value;
    auto status = parse_link(link_value, type_of_value, name_of_value);
    if (status.ok()) {
      object_id = VYObjectIDFromString(name_of_value);
      return Status::OK();
    }
  }
  return Status::Invalid();
}

}  // namespace meta_tree

}  // namespace vineyard
