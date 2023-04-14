/** Copyright 2020-2023 Alibaba Group Holding Limited.

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
#include "arrow/io/api.h"

#include "common/util/json.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/error.h"

namespace vineyard {

class MaxGraphSchema;

using PropertyType = std::shared_ptr<arrow::DataType>;

class Entry {
 public:
  using LabelId = int;
  using PropertyId = int;
  struct PropertyDef {
    PropertyId id;
    std::string name;
    PropertyType type;

    json ToJSON() const;
    void FromJSON(const json& root);
  };
  LabelId id;
  std::string label;
  std::string type;
  std::vector<PropertyDef> props_;
  std::vector<std::string> primary_keys;
  std::vector<std::pair<std::string, std::string>> relations;
  std::vector<int> valid_properties;

  std::vector<int> mapping;          // old prop id -> new prop id
  std::vector<int> reverse_mapping;  // new prop id -> old prop id

  void AddProperty(const std::string& name, PropertyType type);
  void RemoveProperty(const std::string& name);
  void RemoveProperty(const size_t index);

  void AddPrimaryKey(const std::string& key_name);
  void AddPrimaryKeys(const std::vector<std::string>& key_name_list);
  void AddPrimaryKeys(size_t key_count,
                      const std::vector<std::string>& key_name_list);
  void AddRelation(const std::string& src, const std::string& dst);

  size_t property_num() const;

  std::vector<PropertyDef> properties() const;

  PropertyId GetPropertyId(const std::string& name) const;
  PropertyType GetPropertyType(PropertyId prop_id) const;
  std::string GetPropertyName(PropertyId prop_id) const;

  json ToJSON() const;
  void FromJSON(const json& root);

  void InvalidateProperty(PropertyId id) { valid_properties[id] = 0; }
};

class PropertyGraphSchema {
 public:
  using LabelId = int;
  using PropertyId = int;

  PropertyGraphSchema() = default;

  explicit PropertyGraphSchema(json const& json) { FromJSON(json); }

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

  void AddEntry(const Entry& entry);

  const Entry& GetEntry(LabelId label_id, const std::string& type) const;

  Entry& GetMutableEntry(const std::string& label, const std::string& type);

  Entry& GetMutableEntry(const LabelId label_id, const std::string& type);

  json ToJSON() const;
  void ToJSON(json& root) const;
  void FromJSON(json const& root);

  std::string ToJSONString() const;
  void FromJSONString(std::string const& schema);

  void set_fnum(size_t fnum) { fnum_ = fnum; }
  size_t fnum() const { return fnum_; }

  /**
   * N.B.: only valid ones.
   */
  std::vector<Entry> vertex_entries() const;

  /**
   * N.B.: only valid ones.
   */
  std::vector<Entry> edge_entries() const;

  std::vector<Entry> AllVertexEntries() const;

  std::vector<Entry> AllEdgeEntries() const;

  std::vector<Entry> ValidVertexEntries() const;

  std::vector<Entry> ValidEdgeEntries() const;

  bool IsVertexValid(const LabelId label_id) const;

  bool IsEdgeValid(const LabelId label_id) const;

  std::vector<std::string> GetVertexLabels() const;

  std::vector<std::string> GetEdgeLabels() const;

  std::vector<std::pair<std::string, std::string>> GetVertexPropertyListByLabel(
      const std::string& label) const;
  std::vector<std::pair<std::string, std::string>> GetVertexPropertyListByLabel(
      LabelId label_id) const;

  std::vector<std::pair<std::string, std::string>> GetEdgePropertyListByLabel(
      const std::string& label) const;
  std::vector<std::pair<std::string, std::string>> GetEdgePropertyListByLabel(
      LabelId label_id) const;

  bool Validate(std::string& message);

  const std::map<std::string, int>& GetPropertyNameToIDMapping() const;

  void DumpToFile(std::string const& path);

  void InvalidateVertex(LabelId label_id) { valid_vertices_[label_id] = 0; }

  void InvalidateEdge(LabelId label_id) { valid_edges_[label_id] = 0; }

  size_t vertex_label_num() const {
    return std::accumulate(valid_vertices_.begin(), valid_vertices_.end(), 0);
  }

  size_t edge_label_num() const {
    return std::accumulate(valid_edges_.begin(), valid_edges_.end(), 0);
  }

  // For internal use, get all vertex label number include invalid ones.
  size_t all_vertex_label_num() const { return vertex_entries_.size(); }

  // For internal use, get all edge label number include invalid ones.
  size_t all_edge_label_num() const { return edge_entries_.size(); }

  friend MaxGraphSchema;

 public:
  static const std::string VERTEX_TYPE_NAME;
  static const std::string EDGE_TYPE_NAME;

 private:
  size_t fnum_;
  std::vector<Entry> vertex_entries_;
  std::vector<Entry> edge_entries_;
  std::vector<int> valid_vertices_;
  std::vector<int> valid_edges_;
  std::map<std::string, int> name_to_idx_;
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

  void ToJSON(json& root) const;
  void FromJSON(json const& root);

  std::string ToJSONString() const;
  void FromJSONString(std::string const& schema);

  size_t fnum() const { return fnum_; }

  void DumpToFile(std::string const& path);

 private:
  size_t fnum_;
  std::vector<Entry> entries_;
  std::vector<std::string> unique_property_names_;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_GRAPH_SCHEMA_H_
