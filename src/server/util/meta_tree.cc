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

static Status get_sub_tree(const ptree& tree, const std::string& prefix,
                           const std::string& name, ptree& sub_tree) {
  if (name.find('.') != std::string::npos) {
    LOG(ERROR) << "meta tree name invalid. " << name;
    return Status::MetaTreeNameInvalid();
  }
  std::string path = prefix;
  if (!name.empty()) {
    path += "." + name;
  }
  boost::optional<const ptree&> tmp_tree_op = tree.get_child_optional(path);
  if (tmp_tree_op && !tmp_tree_op->empty()) {
    sub_tree = *tmp_tree_op;
    return Status::OK();
  } else {
    return Status::MetaTreeSubtreeNotExists();
  }
}

static bool has_sub_tree(const ptree& tree, const std::string& prefix,
                         const std::string& name) {
  if (name.find('.') != std::string::npos) {
    return false;
  }
  std::string path = prefix;
  if (!name.empty()) {
    path += "." + name;
  }
  return static_cast<bool>(tree.get_child_optional(path));
}

static Status del_sub_tree(ptree& tree, const std::string& prefix,
                           const std::string& name) {
  if (name.find('.') != std::string::npos) {
    LOG(ERROR) << "meta tree name invalid. " << name;
    return Status::MetaTreeNameInvalid();
  }
  std::string path = prefix;
  boost::optional<ptree&> tmp_tree_op = tree.get_child_optional(path);
  if (tmp_tree_op) {
    boost::optional<ptree&> node_op = tmp_tree_op->get_child_optional(name);
    if (node_op) {
      tmp_tree_op->erase(name);
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

static Status get_name(const ptree& tree, std::string& name) {
  // name: get the object id
  ptree::const_assoc_iterator name_iter = tree.find("id");
  if (name_iter == tree.not_found()) {
    return Status::MetaTreeNameNotExists();
  }
  if (!name_iter->second.empty()) {
    LOG(ERROR) << "meta tree id invalid. " << name_iter->second.data();
    return Status::MetaTreeTypeInvalid();
  }
  name = name_iter->second.data();
  return Status::OK();
}

static Status get_type(const ptree& tree, std::string& type,
                       bool const decode = false) {
  // type: get the typename
  ptree::const_assoc_iterator type_iter = tree.find("typename");
  if (type_iter == tree.not_found()) {
    return Status::MetaTreeNameNotExists();
  }
  if (!type_iter->second.empty()) {
    LOG(ERROR) << "meta tree typename invalid. " << type_iter->second.data();
    return Status::MetaTreeTypeInvalid();
  }
  type = type_iter->second.data();
  if (decode) {
    NodeType node_type = NodeType::InvalidType;
    decode_value(type, node_type, type);
    if (node_type != NodeType::Value) {
      return Status::MetaTreeTypeInvalid();
    }
  }
  return Status::OK();
}

static Status get_type_name(const ptree& tree, std::string& type,
                            std::string& name) {
  RETURN_ON_ERROR(get_type(tree, type));
  RETURN_ON_ERROR(get_name(tree, name));
  return Status::OK();
}

/**
 * In `MetaData::AddMember`, the parameter might be an object id. In such cases
 * the client doesn't have the full metadata ptree of the object, there will be
 * just an `ObjectID`.
 */
static bool is_meta_placeholder(const ptree& tree) {
  return tree.size() == 1 && tree.find("id") != tree.not_found();
}

/**
 * Get metadata for an object "recursively".
 */
Status GetData(const ptree& tree, const ObjectID id, ptree& sub_tree) {
  return GetData(tree, VYObjectIDToString(id), sub_tree);
}

/**
 * Get metadata for an object "recursively".
 */
Status GetData(const ptree& tree, const std::string& name, ptree& sub_tree) {
  ptree tmp_tree;
  sub_tree.clear();
  Status status = get_sub_tree(tree, "data", name, tmp_tree);
  if (!status.ok()) {
    return status;
  }
  for (ptree::iterator it = tmp_tree.begin(); it != tmp_tree.end(); ++it) {
    NodeType type;
    std::string value;
    decode_value(it->second.data(), type, value);
    if (type == NodeType::Value) {
      sub_tree.put(it->first, value);
    } else if (type == NodeType::Link) {
      std::string sub_sub_tree_type, sub_sub_tree_name;
      status = parse_link(value, sub_sub_tree_type, sub_sub_tree_name);
      if (!status.ok()) {
        sub_tree.clear();
        return status;
      }
      ptree sub_sub_tree;
      status =
          GetData(tree, VYObjectIDFromString(sub_sub_tree_name), sub_sub_tree);
      if (!status.ok()) {
        sub_tree.clear();
        return status;
      }
      sub_tree.add_child(it->first, sub_sub_tree);
    } else {
      return Status::MetaTreeTypeInvalid();
    }
  }
  sub_tree.put("id", name);
  return Status::OK();
}

Status ListData(const ptree& tree, std::string const& pattern, bool const regex,
                size_t const limit, ptree& tree_group) {
  auto metas = tree.get_child_optional("data");
  if (!metas) {
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

  for (auto iter = metas->begin(); iter != metas->end(); ++iter) {
    if (found >= limit) {
      break;
    }

    if (iter->second.empty()) {
      LOG(INFO) << "Object meta shouldn't be empty";
      return Status::MetaTreeInvalid();
    }
    std::string type;
    RETURN_ON_ERROR(get_type(iter->second, type, true));

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
      ptree object_meta_tree;
      RETURN_ON_ERROR(GetData(tree, iter->first, object_meta_tree));
      tree_group.add_child(iter->first, object_meta_tree);
    }
  }
  return Status::OK();
}

Status DelData(ptree& tree, const ObjectID id) {
  std::string name = VYObjectIDToString(id);
  return del_sub_tree(tree, "data", name);
}

Status DelData(ptree& tree, const std::vector<ObjectID>& ids) {
  // FIXME: use a more efficient implmentation.
  for (auto const& id : ids) {
    auto s = DelData(tree, id);
    if (!s.ok()) {
      return s;
    }
  }
  return Status::OK();
}

Status DelDataOps(const ptree& tree, const ObjectID id,
                  std::vector<IMetaService::op_t>& ops) {
  return DelDataOps(tree, VYObjectIDToString(id), ops);
}

Status DelDataOps(const ptree& tree, const std::set<ObjectID>& ids,
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

Status DelDataOps(const ptree& tree, const std::string& name,
                  std::vector<IMetaService::op_t>& ops) {
  std::string data_prefix = "data";
  boost::optional<const ptree&> data_tree_optional =
      tree.get_child_optional(data_prefix);
  if (data_tree_optional) {
    boost::optional<const ptree&> node_op =
        data_tree_optional->get_child_optional(name);
    if (node_op) {
      // erase from etcd
      std::string key_prefix = data_prefix + "." + name + ".";
      for (auto it = node_op->begin(); it != node_op->end(); ++it) {
        ops.emplace_back(IMetaService::op_t::Del(key_prefix + it->first));
      }
      // ensure the node will be erased from in-server ptree correctly.
      ops.emplace_back(IMetaService::op_t::Del(data_prefix + "." + name));
      return Status::OK();
    }
  }
  return Status::MetaTreeSubtreeNotExists();
}

static void generate_put_ops(const ptree& meta, const ptree& diff,
                             const std::string& name,
                             std::vector<IMetaService::op_t>& ops) {
  std::string key_prefix = "data." + name + ".";
  for (ptree::const_iterator it = diff.begin(); it != diff.end(); ++it) {
    if (!it->second.empty()) {
      std::string sub_type, sub_name;
      VINEYARD_SUPPRESS(get_type_name(it->second, sub_type, sub_name));
      if (!has_sub_tree(meta, "data", name)) {
        generate_put_ops(meta, it->second, sub_name, ops);
      }
      std::string link;
      generate_link(sub_type, sub_name, link);
      std::string encoded_value;
      encode_value(NodeType::Link, link, encoded_value);
      ops.emplace_back(
          IMetaService::op_t::Put(key_prefix + it->first, encoded_value));
    } else {
      // don't repeat "id" in the etcd kvs.
      if (it->first == "id") {
        continue;
      }
      std::string encoded_value;
      encode_value(NodeType::Value, it->second.data(), encoded_value);
      ops.emplace_back(
          IMetaService::op_t::Put(key_prefix + it->first, encoded_value));
    }
  }
}

static void generate_persist_ops(const ptree& diff, const std::string& name,
                                 std::vector<IMetaService::op_t>& ops,
                                 std::set<std::string>& dedup) {
  std::string key_prefix = "data." + name + ".";
  for (ptree::const_iterator it = diff.begin(); it != diff.end(); ++it) {
    if (!it->second.empty()) {
      std::string sub_type, sub_name;
      VINEYARD_SUPPRESS(get_type_name(it->second, sub_type, sub_name));
      if (it->second.get<bool>("transient")) {
        // otherwise, skip recursively generate ops
        generate_persist_ops(it->second, sub_name, ops, dedup);
      }
      std::string link;
      generate_link(sub_type, sub_name, link);
      std::string encoded_value;
      encode_value(NodeType::Link, link, encoded_value);
      std::string encoded_key = key_prefix + it->first;
      if (dedup.find(encoded_key) == dedup.end()) {
        ops.emplace_back(IMetaService::op_t::Put(encoded_key, encoded_value));
        dedup.emplace(encoded_key);
      }
    } else {
      // don't repeat "id" in the etcd kvs.
      if (it->first == "id") {
        continue;
      }
      std::string encoded_value;
      if (it->first == "transient") {
        encode_value(NodeType::Value, "false", encoded_value);
      } else {
        encode_value(NodeType::Value, it->second.data(), encoded_value);
      }
      std::string encoded_key = key_prefix + it->first;
      if (dedup.find(encoded_key) == dedup.end()) {
        ops.emplace_back(IMetaService::op_t::Put(encoded_key, encoded_value));
        dedup.emplace(encoded_key);
      }
    }
  }
}

/**
 * Returns:
 *
 *  diff: diff ptree
 *  instance_id: instance_id of members and the object itself, can represents
 *               the final instance_id of the object.
 */
static Status diff_data_meta_tree(const ptree& meta,
                                  const std::string& sub_tree_name,
                                  const ptree& sub_tree, ptree& diff,
                                  InstanceID& instance_id) {
  ptree old_sub_tree;
  Status status = get_sub_tree(meta, "data", sub_tree_name, old_sub_tree);

  if (!status.ok()) {
    if (status.IsMetaTreeSubtreeNotExists()) {
      diff.put("transient", true);
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
      diff.put("id", sub_tree_name);
      diff.put("typename", sub_tree_type);
      {
        const std::string& instance_id_string =
            old_sub_tree.get<std::string>("instance_id");
        NodeType instance_id_value_type;
        std::string instance_id_decoded;
        decode_value(instance_id_string, instance_id_value_type,
                     instance_id_decoded);
        instance_id = boost::lexical_cast<InstanceID>(instance_id_decoded);
      }
    }
    return status;
  }

  // recompute instance_id: check if it refers remote objects.
  instance_id = sub_tree.get<InstanceID>("instance_id");

  for (ptree::const_iterator it = sub_tree.begin(); it != sub_tree.end();
       ++it) {
    // don't diff on id, typename and instance_id and don't update them, that
    // means, we can only update member's meta, cannot update the whole member
    // itself.
    if (it->first == "id" || it->first == "typename" ||
        it->first == "instance_id") {
      continue;
    }

    if (it->second.empty() /* plain value */) {
      std::string new_value = it->second.data();
      if (status.ok() /* old meta exists */) {
        boost::optional<std::string> old_value =
            old_sub_tree.get_optional<std::string>(it->first);
        if (old_value) {
          NodeType old_value_type;
          std::string old_value_decoded;
          decode_value(*old_value, old_value_type, old_value_decoded);

          bool require_update = false;
          if (it->first == "transient") {
            // put_data wan't make persist value becomes transient, since the
            // info in the client may be out-of-date.
            require_update =
                old_value_decoded == "true" && old_value_decoded != new_value;
          } else {
            require_update = old_value_decoded != new_value;
          }

          if (require_update) {
            VLOG(10) << "DIFF: " << it->first << ": " << old_value_decoded
                     << " -> " << new_value;
            diff.put(it->first, new_value);
          }
        } else {
          VLOG(10) << "DIFF: " << it->first << ": [none] -> " << new_value;
          diff.put(it->first, new_value);
        }
      } else if (status.IsMetaTreeSubtreeNotExists()) {
        VLOG(10) << "DIFF: " << it->first << ": [none] -> " << new_value;
        diff.put(it->first, new_value);
      } else {
        return status;
      }
    } else /* member object */ {
      const ptree& sub_sub_tree = it->second;

      // original corresponding field must be a member not a key-value
      auto mb_old_sub_sub_tree = old_sub_tree.find(it->first);
      if (status.ok() /* old meta exists */) {
        if (mb_old_sub_sub_tree != old_sub_tree.not_found() &&
            !is_link_node(mb_old_sub_sub_tree->second.data())) {
          return Status::MetaTreeInvalid();
        }
      }

      std::string sub_sub_tree_name;
      RETURN_ON_ERROR(get_name(sub_sub_tree, sub_sub_tree_name));

      ptree diff_sub_tree;
      InstanceID sub_instance_id;
      RETURN_ON_ERROR(diff_data_meta_tree(meta, sub_sub_tree_name, sub_sub_tree,
                                          diff_sub_tree, sub_instance_id));

      if (instance_id != sub_instance_id) {
        instance_id = UnspecifiedInstanceID();
      }

      if (status.ok() /* old meta exists */) {
        if (!diff_sub_tree.empty()) {
          diff.add_child(it->first, diff_sub_tree);
        }
      } else if (status.IsMetaTreeSubtreeNotExists()) {
        if (!is_meta_placeholder(sub_sub_tree)) {
          // if sub_sub_tree is placeholder, the diff already contains id and
          // typename
          std::string sub_sub_tree_type;
          RETURN_ON_ERROR(get_type_name(sub_sub_tree, sub_sub_tree_type,
                                        sub_sub_tree_name));
          diff_sub_tree.put("id", sub_sub_tree_name);
          diff_sub_tree.put("typename", sub_sub_tree_type);
        }
        diff.add_child(it->first, diff_sub_tree);
      } else {
        return status;
      }
    }
  }
  if (!diff.empty() && status.IsMetaTreeSubtreeNotExists()) {
    // must not be meta placeholder
    std::string sub_tree_type;
    RETURN_ON_ERROR(get_type(sub_tree, sub_tree_type));

    diff.put("id", sub_tree_name);
    diff.put("typename", sub_tree_type);
    diff.put("instance_id", instance_id);
  }
  return Status::OK();
}

static bool persist_meta_tree(const ptree& sub_tree, ptree& diff) {
  // NB: we don't need to track which objects are persist since the ptree
  // cached in the server will be updated by the background watcher task.
  if (IsBlob(VYObjectIDFromString(sub_tree.get<std::string>("id")))) {
    // Don't persist blob into etcd.
    return false;
  }
  if (sub_tree.get<bool>("transient")) {
    for (ptree::const_iterator it = sub_tree.begin(); it != sub_tree.end();
         ++it) {
      if (!it->second.empty()) {
        const ptree& sub_sub_tree = it->second;
        // recursive
        ptree sub_diff;
        bool ret = persist_meta_tree(sub_sub_tree, sub_diff);
        if (!sub_diff.empty()) {
          diff.add_child(it->first, sub_diff);
        } else if (ret) {
          // will be used to generate the link.
          diff.add_child(it->first, sub_sub_tree);
        }
      } else {
        diff.put(it->first, it->second.data());
      }
    }
  }
  return true;
}

Status PutDataOps(const ptree& tree, const ObjectID id, const ptree& sub_tree,
                  std::vector<IMetaService::op_t>& ops,
                  InstanceID& computed_instance_id) {
  ptree diff;
  std::string name = VYObjectIDToString(id);
  // recompute instance_id: check if it refers remote objects.
  Status status =
      diff_data_meta_tree(tree, name, sub_tree, diff, computed_instance_id);

  if (!status.ok()) {
    return status;
  }

  if (diff.empty()) {
    return Status::OK();
  }

  generate_put_ops(tree, diff, name, ops);
  return Status::OK();
}

Status PersistOps(const ptree& tree, const ObjectID id,
                  std::vector<IMetaService::op_t>& ops) {
  ptree sub_tree, diff;
  Status status = GetData(tree, id, sub_tree);
  if (!status.ok()) {
    return status;
  }
  persist_meta_tree(sub_tree, diff);

  if (diff.empty()) {
    return Status::OK();
  }

  std::string name = VYObjectIDToString(id);
  std::set<std::string> dedup;
  generate_persist_ops(diff, name, ops, dedup);
  return Status::OK();
}

Status Exists(const ptree& tree, const ObjectID id, bool& exists) {
  std::string name = VYObjectIDToString(id);
  exists = has_sub_tree(tree, "data", name);
  return Status::OK();
}

Status ShallowCopyOps(const ptree& tree, const ObjectID id,
                      const ObjectID target,
                      std::vector<IMetaService::op_t>& ops, bool& transient) {
  std::string name = VYObjectIDToString(id);
  ptree tmp_tree;
  RETURN_ON_ERROR(get_sub_tree(tree, "data", name, tmp_tree));
  NodeType field_type;
  std::string field_value;
  decode_value(tmp_tree.get<std::string>("transient"), field_type, field_value);
  RETURN_ON_ASSERT(field_type == NodeType::Value,
                   "The 'transient' should a plain value");
  transient = boost::lexical_cast<bool>(field_value);
  std::string key_prefix = "data." + VYObjectIDToString(target) + ".";
  for (auto const& kv : tmp_tree) {
    ops.emplace_back(
        IMetaService::op_t::Put(key_prefix + kv.first, kv.second.data()));
  }
  return Status::OK();
}

Status IfPersist(const ptree& tree, const ObjectID id, bool& persist) {
  std::string name = VYObjectIDToString(id);
  ptree tmp_tree;
  Status status = get_sub_tree(tree, "data", name, tmp_tree);
  if (status.ok()) {
    NodeType field_type;
    std::string field_value;
    decode_value(tmp_tree.get<std::string>("transient"), field_type,
                 field_value);
    RETURN_ON_ASSERT(field_type == NodeType::Value,
                     "The 'transient' should a plain value");
    persist = !boost::lexical_cast<bool>(field_value);
  }
  return status;
}

Status FilterAtInstance(const ptree& tree, const InstanceID& instance_id,
                        std::vector<ObjectID>& objects) {
  std::string instance_id_value;
  encode_value(NodeType::Value, std::to_string(instance_id), instance_id_value);
  auto mb_datatree = tree.get_child_optional("data");
  if (mb_datatree) {
    auto& datatree = mb_datatree.get();
    for (auto it = datatree.begin(); it != datatree.end(); ++it) {
      if (!it->second.empty()) {
        auto mb_instance_id =
            it->second.get_optional<std::string>("instance_id");
        if (mb_instance_id && mb_instance_id.get() == instance_id_value) {
          objects.emplace_back(VYObjectIDFromString(it->first));
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
