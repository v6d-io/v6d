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

#include "graph/fragment/graph_schema.h"

#include <cctype>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "arrow/api.h"
#include "glog/logging.h"

#include "common/util/ptree.h"

namespace vineyard {

namespace detail {

std::string PropertyTypeToString(PropertyType type) {
  if (arrow::boolean()->Equals(type)) {
    return "BOOL";
  } else if (arrow::int16()->Equals(type)) {
    return "SHORT";
  } else if (arrow::int32()->Equals(type)) {
    return "INT";
  } else if (arrow::int64()->Equals(type)) {
    return "LONG";
  } else if (arrow::float32()->Equals(type)) {
    return "FLOAT";
  } else if (arrow::float64()->Equals(type)) {
    return "DOUBLE";
  } else if (arrow::utf8()->Equals(type)) {
    return "STRING";
  }
  LOG(ERROR) << "Unsupported arrow type " << type->ToString();
  return "NULL";
}

std::string toupper(const std::string& s) {
  std::string upper_s = s;
  std::transform(s.begin(), s.end(), upper_s.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return upper_s;
}

PropertyType PropertyTypeFromString(const std::string& type) {
  auto type_upper = toupper(type);
  if (type_upper == "BOOL") {
    return arrow::boolean();
  } else if (type_upper == "SHORT") {
    return arrow::int16();
  } else if (type_upper == "INT") {
    return arrow::int32();
  } else if (type_upper == "LONG") {
    return arrow::int64();
  } else if (type_upper == "FLOAT") {
    return arrow::float32();
  } else if (type_upper == "DOUBLE") {
    return arrow::float64();
  } else if (type_upper == "STRING") {
    return arrow::utf8();
  } else {
    LOG(ERROR) << "Unsupported property type " << type;
  }
  return arrow::null();
}

}  // namespace detail

using boost::property_tree::ptree;

ptree Entry::PropertyDef::ToJSON() const {
  ptree root;
  root.put("id", id);
  root.put("name", name);
  auto type_str = detail::PropertyTypeToString(type);
  root.put("data_type", type_str);
  return root;
}

void Entry::PropertyDef::FromJSON(const ptree& root) {
  id = root.get<PropertyId>("id");
  name = root.get<std::string>("name");
  type = detail::PropertyTypeFromString(root.get<std::string>("data_type"));
}

void Entry::AddProperty(const std::string& name, PropertyType type) {
  props.emplace_back(PropertyDef{
      .id = static_cast<int>(props.size()), .name = name, .type = type});
}

void Entry::AddPrimaryKeys(size_t key_count,
                           const std::vector<std::string>& key_name_list) {
  for (size_t idx = 0; idx < key_count; ++idx) {
    primary_keys.emplace_back(key_name_list[idx]);
  }
}

void Entry::AddRelation(const std::string& src, const std::string& dst) {
  relations.emplace_back(src, dst);
}

Entry::PropertyId Entry::GetPropertyId(const std::string& name) const {
  for (const auto& prop : props) {
    if (prop.name == name) {
      return prop.id;
    }
  }
  return -1;
}

std::string Entry::GetPropertyName(PropertyId prop_id) const {
  for (const auto& prop : props) {
    if (prop.id == prop_id) {
      return prop.name;
    }
  }
  return "";
}

PropertyType Entry::GetPropertyType(PropertyId prop_id) const {
  for (const auto& prop : props) {
    if (prop.id == prop_id) {
      return prop.type;
    }
  }
  return arrow::null();
}

ptree Entry::ToJSON() const {
  ptree root;
  root.put("id", id);
  root.put("label", label);
  root.put("type", type);
  ptree prop_array, pk_array, relation_array;
  // propertyDefList
  for (const auto& prop : props) {
    prop_array.push_back(std::make_pair("", prop.ToJSON()));
  }
  root.add_child("propertyDefList", prop_array);
  // indexes
  if (!primary_keys.empty()) {
    ptree index_array, pk_array_tree;
    for (const auto& pk : primary_keys) {
      ptree pk_tree;
      pk_tree.put("", pk);
      pk_array.push_back(std::make_pair("", pk_tree));
    }
    pk_array_tree.add_child("propertyNames", pk_array);
    index_array.push_back(std::make_pair("", pk_array_tree));
    root.add_child("indexes", index_array);
  }
  // rawRelationShips
  if (!relations.empty()) {
    for (const auto& rel : relations) {
      ptree edge_tree;
      edge_tree.put("srcVertexLabel", rel.first);
      edge_tree.put("dstVertexLabel", rel.second);
      relation_array.push_back(std::make_pair("", edge_tree));
    }
    root.add_child("rawRelationShips", relation_array);
  }
  // mappings
  if (!mapping.empty()) {
    vineyard::put_container(root, "mapping", mapping);
  }
  if (!reverse_mapping.empty()) {
    vineyard::put_container(root, "reverse_mapping", reverse_mapping);
  }
  return root;
}

void Entry::FromJSON(const ptree& root) {
  id = root.get<LabelId>("id");
  label = root.get<std::string>("label");
  type = root.get<std::string>("type");
  // propertyDefList
  const ptree& prop_array = root.get_child("propertyDefList");
  for (const auto& kv : prop_array) {
    PropertyDef prop;
    prop.FromJSON(kv.second);
    props.emplace_back(prop);
  }
  // indexes
  if (root.get_child_optional("indexes")) {
    for (const auto& index_arr_kv : root.get_child("indexes")) {
      auto pk_arr = index_arr_kv.second.get_child_optional("propertyNames");
      if (pk_arr) {
        for (const auto& kv : pk_arr.get()) {
          primary_keys.emplace_back(kv.second.data());
        }
        break;
      }
    }
  }
  // rawRelationShips
  if (root.get_child_optional("rawRelationShips")) {
    for (const auto& index_arr_kv : root.get_child("rawRelationShips")) {
      auto src =
          index_arr_kv.second.get_optional<std::string>("srcVertexLabel");
      auto dst =
          index_arr_kv.second.get_optional<std::string>("dstVertexLabel");
      if (src && dst) {
        relations.emplace_back(src.get(), dst.get());
      }
    }
  }
  // mapping
  if (root.get_optional<std::string>("mapping")) {
    vineyard::get_container(root, "mapping", mapping);
  }
  if (root.get_optional<std::string>("reverse_mapping")) {
    vineyard::get_container(root, "reverse_mapping", reverse_mapping);
  }
}

PropertyGraphSchema::PropertyId PropertyGraphSchema::GetVertexPropertyId(
    LabelId label_id, const std::string& name) const {
  return vertex_entries_[label_id].GetPropertyId(name);
}

PropertyType PropertyGraphSchema::GetVertexPropertyType(
    LabelId label_id, PropertyId prop_id) const {
  return vertex_entries_[label_id].props[prop_id].type;
}

std::string PropertyGraphSchema::GetVertexPropertyName(
    LabelId label_id, PropertyId prop_id) const {
  return vertex_entries_[label_id].props[prop_id].name;
}

PropertyGraphSchema::PropertyId PropertyGraphSchema::GetEdgePropertyId(
    LabelId label_id, const std::string& name) const {
  return edge_entries_[label_id].GetPropertyId(name);
}

PropertyType PropertyGraphSchema::GetEdgePropertyType(
    LabelId label_id, PropertyId prop_id) const {
  return edge_entries_[label_id].props[prop_id].type;
}

std::string PropertyGraphSchema::GetEdgePropertyName(LabelId label_id,
                                                     PropertyId prop_id) const {
  return edge_entries_[label_id].props[prop_id].name;
}

PropertyGraphSchema::LabelId PropertyGraphSchema::GetVertexLabelId(
    const std::string& name) const {
  for (const auto& entry : vertex_entries_) {
    if (entry.label == name) {
      return entry.id;
    }
  }
  return -1;
}

std::string PropertyGraphSchema::GetVertexLabelName(LabelId label_id) const {
  return vertex_entries_[label_id].label;
}

PropertyGraphSchema::LabelId PropertyGraphSchema::GetEdgeLabelId(
    const std::string& name) const {
  for (const auto& entry : edge_entries_) {
    if (entry.label == name) {
      return entry.id;
    }
  }
  return -1;
}

std::string PropertyGraphSchema::GetEdgeLabelName(LabelId label_id) const {
  return vertex_entries_[label_id].label;
}

Entry* PropertyGraphSchema::CreateEntry(const std::string& name,
                                        const std::string& type) {
  if (type == "VERTEX") {
    vertex_entries_.emplace_back(
        Entry{.id = static_cast<int>(vertex_entries_.size()),
              .label = name,
              .type = type});
    return &*vertex_entries_.rbegin();
  } else {
    edge_entries_.emplace_back(
        Entry{.id = static_cast<int>(edge_entries_.size()),
              .label = name,
              .type = type});
    return &*edge_entries_.rbegin();
  }
}

std::vector<std::string> PropertyGraphSchema::GetVextexLabels() const {
  std::vector<std::string> labels;
  for (auto& entry : vertex_entries_) {
    labels.emplace_back(entry.label);
  }
  return labels;
}

std::vector<std::string> PropertyGraphSchema::GetEdgeLabels() const {
  std::vector<std::string> labels;
  for (auto& entry : edge_entries_) {
    labels.emplace_back(entry.label);
  }
  return labels;
}

std::vector<std::pair<std::string, std::string>>
PropertyGraphSchema::GetVertexPropertyListByLabel(
    const std::string& label) const {
  LabelId label_id = GetVertexLabelId(label);
  return GetVertexPropertyListByLabel(label_id);
}

std::vector<std::pair<std::string, std::string>>
PropertyGraphSchema::GetVertexPropertyListByLabel(LabelId label_id) const {
  std::vector<std::pair<std::string, std::string>> properties;
  for (auto& prop : vertex_entries_[label_id].props) {
    properties.emplace_back(prop.name, detail::PropertyTypeToString(prop.type));
  }
  return properties;
}

std::vector<std::pair<std::string, std::string>>
PropertyGraphSchema::GetEdgePropertyListByLabel(
    const std::string& label) const {
  LabelId label_id = GetEdgeLabelId(label);
  return GetEdgePropertyListByLabel(label_id);
}

std::vector<std::pair<std::string, std::string>>
PropertyGraphSchema::GetEdgePropertyListByLabel(LabelId label_id) const {
  std::vector<std::pair<std::string, std::string>> properties;
  for (auto& prop : edge_entries_[label_id].props) {
    properties.emplace_back(prop.name, detail::PropertyTypeToString(prop.type));
  }
  return properties;
}

void PropertyGraphSchema::ToJSON(ptree& root) const {
  root.put("partitionNum", fnum_);
  ptree types;
  for (const auto& entry : vertex_entries_) {
    types.push_back(std::make_pair("", entry.ToJSON()));
  }
  for (const auto& entry : edge_entries_) {
    types.push_back(std::make_pair("", entry.ToJSON()));
  }
  root.add_child("types", types);
}

void PropertyGraphSchema::FromJSON(ptree const& root) {
  fnum_ = root.get<size_t>("partitionNum");
  for (const auto& kv : root.get_child("types")) {
    Entry entry;
    entry.FromJSON(kv.second);
    if (entry.type == "VERTEX") {
      vertex_entries_.push_back(std::move(entry));
    } else {
      edge_entries_.push_back(std::move(entry));
    }
  }
}

std::string PropertyGraphSchema::ToJSONString() const {
  std::stringstream ss;
  ptree root;
  ToJSON(root);
  boost::property_tree::write_json(ss, root, false);
  return ss.str();
}

void PropertyGraphSchema::FromJSONString(std::string const& schema) {
  ptree root;
  std::istringstream iss(schema);
  boost::property_tree::read_json(iss, root);
  FromJSON(root);
}

void PropertyGraphSchema::DumpToFile(std::string const& path) {
  std::ofstream json_file;
  json_file.open(path);
  json_file << this->ToJSONString();
  json_file.close();
}

MaxGraphSchema::MaxGraphSchema(const PropertyGraphSchema& schema) {
  const auto& v_entries = schema.vertex_entries();
  const auto& e_entries = schema.edge_entries();
  // Gather all property names and unique them
  std::set<std::string> prop_names;
  for (const auto& entry : v_entries) {
    for (const auto& prop : entry.props) {
      prop_names.insert(prop.name);
    }
  }
  for (const auto& entry : e_entries) {
    for (const auto& prop : entry.props) {
      prop_names.insert(prop.name);
    }
  }

  // Assign a id to each name.
  std::map<std::string, int> name_to_idx;
  for (auto iter = prop_names.begin(); iter != prop_names.end(); ++iter) {
    name_to_idx[*iter] = std::distance(prop_names.begin(), iter);
  }

  // Assign generated id to property by name.
  for (const auto& entry : v_entries) {
    Entry new_entry = entry;
    for (auto& prop : new_entry.props) {
      new_entry.mapping[prop.id] = name_to_idx[prop.name];
      new_entry.reverse_mapping[name_to_idx[prop.name]] = prop.id;
      prop.id = name_to_idx[prop.name];
    }
    entries_.push_back(new_entry);
  }
  int vertex_label_num = v_entries.size();
  for (const auto& entry : e_entries) {
    Entry new_entry = entry;
    new_entry.id += vertex_label_num;
    for (auto& prop : new_entry.props) {
      new_entry.mapping[prop.id] = name_to_idx[prop.name];
      new_entry.reverse_mapping[name_to_idx[prop.name]] = prop.id;
      prop.id = name_to_idx[prop.name];
    }
    entries_.push_back(new_entry);
  }
  fnum_ = schema.fnum();
}

MaxGraphSchema::PropertyId MaxGraphSchema::GetPropertyId(
    const std::string& name) {
  PropertyId id;
  for (const auto& entry : entries_) {
    id = entry.GetPropertyId(name);
    if (id != -1) {
      return id;
    }
  }
  return -1;
}

PropertyType MaxGraphSchema::GetPropertyType(LabelId label_id,
                                             PropertyId prop_id) {
  PropertyType type;
  for (const auto& entry : entries_) {
    if (entry.id == label_id) {
      type = entry.GetPropertyType(prop_id);
      if (!type->Equals(arrow::null())) {
        return type;
      }
    }
  }
  return arrow::null();
}

std::string MaxGraphSchema::GetPropertyName(PropertyId prop_id) {
  std::string name;
  for (const auto& entry : entries_) {
    name = entry.GetPropertyName(prop_id);
    if (!name.empty()) {
      return name;
    }
  }
  return "";
}

MaxGraphSchema::LabelId MaxGraphSchema::GetLabelId(const std::string& name) {
  for (const auto& entry : entries_) {
    if (entry.label == name) {
      return entry.id;
    }
  }
  return -1;
}

std::string MaxGraphSchema::GetLabelName(LabelId label_id) {
  for (const auto& entry : entries_) {
    if (entry.id == label_id) {
      return entry.label;
    }
  }
  return "";
}

void MaxGraphSchema::ToJSON(ptree& root) const {
  root.put("partitionNum", fnum_);
  ptree types;
  for (const auto& entry : entries_) {
    types.push_back(std::make_pair("", entry.ToJSON()));
  }
  root.add_child("types", types);
}

void MaxGraphSchema::FromJSON(ptree const& root) {
  fnum_ = root.get<size_t>("partitionNum");
  for (const auto& kv : root.get_child("types")) {
    Entry entry;
    entry.FromJSON(kv.second);
    entries_.push_back(std::move(entry));
  }
}

std::string MaxGraphSchema::ToJSONString() const {
  std::stringstream ss;
  ptree root;
  ToJSON(root);
  boost::property_tree::write_json(ss, root, false);
  return ss.str();
}

void MaxGraphSchema::FromJSONString(std::string const& schema) {
  ptree root;
  std::istringstream iss(schema);
  boost::property_tree::read_json(iss, root);
  FromJSON(root);
}

void MaxGraphSchema::DumpToFile(std::string const& path) {
  std::ofstream json_file;
  json_file.open(path);
  json_file << this->ToJSONString();
  json_file.close();
}

}  // namespace vineyard
