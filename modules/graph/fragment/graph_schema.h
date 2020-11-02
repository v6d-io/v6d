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

#ifndef MODULES_GRAPH_FRAGMENT_GRAPH_SCHEMA_H_
#define MODULES_GRAPH_FRAGMENT_GRAPH_SCHEMA_H_

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "boost/leaf/all.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/property_tree/ptree.hpp"

#include "graph/fragment/property_graph_types.h"

namespace vineyard {

using PropertyType = std::shared_ptr<arrow::DataType>;
class Entry {
 public:
  using LabelId = int;
  using PropertyId = int;
  struct PropertyDef {
    PropertyId id;
    std::string name;
    PropertyType type;

    boost::property_tree::ptree ToJSON() const;
    void FromJSON(const boost::property_tree::ptree& root);
  };
  LabelId id;
  std::string label;
  std::string type;
  std::vector<PropertyDef> props;
  std::vector<std::string> primary_keys;
  std::vector<std::pair<std::string, std::string>> relations;

  std::vector<int> mapping;          // old prop id -> new prop id
  std::vector<int> reverse_mapping;  // new prop id -> old prop id

  void AddProperty(const std::string& name, PropertyType type);
  void AddPrimaryKeys(size_t key_count,
                      const std::vector<std::string>& key_name_list);
  void AddRelation(const std::string& src, const std::string& dst);

  PropertyId GetPropertyId(const std::string& name) const;
  PropertyType GetPropertyType(PropertyId prop_id) const;
  std::string GetPropertyName(PropertyId prop_id) const;

  boost::property_tree::ptree ToJSON() const;
  void FromJSON(const boost::property_tree::ptree& root);
};

class PropertyGraphSchema {
 public:
  using LabelId = int;
  using PropertyId = int;

  PropertyId GetVertexPropertyId(LabelId label_id,
                                 const std::string& name) const;
  PropertyType GetVertexPropertyType(LabelId label_id,
                                     PropertyId prop_id) const;
  std::string GetVertexPropertyName(LabelId label_id, PropertyId prop_id) const;

  PropertyId GetEdgePropertyId(LabelId label_id, const std::string& name) const;
  PropertyType GetEdgePropertyType(LabelId label_id, PropertyId prop_id) const;
  std::string GetEdgePropertyName(LabelId label_id, PropertyId prop_id) const;

  LabelId GetVertexLabelId(const std::string& name) const;
  std::string GetVertexLabelName(LabelId label_id) const;

  LabelId GetEdgeLabelId(const std::string& name) const;
  std::string GetEdgeLabelName(LabelId label_id) const;

  Entry* CreateEntry(const std::string& name, const std::string& type);

  void AddEntry(const Entry& entry) {
    if (entry.type == "VERTEX") {
      vertex_entries_.push_back(entry);
    } else {
      edge_entries_.push_back(entry);
    }
  }

  Entry& GetMutableEntry(const std::string& label, const std::string& type) {
    if (type == "VERTEX") {
      for (auto& entry : vertex_entries_) {
        if (entry.label == label) {
          return entry;
        }
      }
    } else {
      for (auto& entry : edge_entries_) {
        if (entry.label == label) {
          return entry;
        }
      }
    }
    throw std::runtime_error("Not found the entry of label " + type + " " +
                             label);
  }

  void ToJSON(boost::property_tree::ptree& root) const;
  void FromJSON(boost::property_tree::ptree const& root);

  std::string ToJSONString() const;
  void FromJSONString(std::string const& schema);

  void set_fnum(size_t fnum) { fnum_ = fnum; }
  size_t fnum() const { return fnum_; }

  const std::vector<Entry>& vertex_entries() const { return vertex_entries_; }

  const std::vector<Entry>& edge_entries() const { return edge_entries_; }

  std::vector<std::string> GetVextexLabels() const;

  std::vector<std::string> GetEdgeLabels() const;

  std::vector<std::pair<std::string, std::string>> GetVertexPropertyListByLabel(
      const std::string& label) const;
  std::vector<std::pair<std::string, std::string>> GetVertexPropertyListByLabel(
      LabelId label_id) const;

  std::vector<std::pair<std::string, std::string>> GetEdgePropertyListByLabel(
      const std::string& label) const;
  std::vector<std::pair<std::string, std::string>> GetEdgePropertyListByLabel(
      LabelId label_id) const;

  void DumpToFile(std::string const& path);

 private:
  size_t fnum_;
  std::vector<Entry> vertex_entries_;
  std::vector<Entry> edge_entries_;
};

// In Analytical engine, assume label ids of vertex entries are continuous
// from zero, and property ids of each label is also continuous from zero.
// When transform schema to Maxgraph style, we gather all property names and
// unique them, assign each name a id (index of the vector), then preserve a
// vector<int> for each label, stores mappings from original id to transformed
// id.
class MaxGraphSchema {
 public:
  using LabelId = int;
  using PropertyId = int;
  explicit MaxGraphSchema(const PropertyGraphSchema& schema);
  PropertyId GetPropertyId(const std::string& name);
  PropertyType GetPropertyType(LabelId label_id, PropertyId prop_id);
  std::string GetPropertyName(PropertyId prop_id);

  LabelId GetLabelId(const std::string& name);
  std::string GetLabelName(LabelId label_id);

  void set_fnum(size_t fnum) { fnum_ = fnum; }

  void AddEntry(const Entry& entry) { entries_.push_back(entry); }

  void ToJSON(boost::property_tree::ptree& root) const;
  void FromJSON(boost::property_tree::ptree const& root);

  std::string ToJSONString() const;
  void FromJSONString(std::string const& schema);

  size_t fnum() const { return fnum_; }

  void DumpToFile(std::string const& path);

 private:
  size_t fnum_;
  std::vector<Entry> entries_;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_GRAPH_SCHEMA_H_
