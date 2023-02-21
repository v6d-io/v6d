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

#ifndef MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GROUP_H_
#define MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GROUP_H_

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "client/client.h"

#include "graph/fragment/property_graph_types.h"

namespace vineyard {

class ArrowFragmentGroupBuilder;

class ArrowFragmentGroup : public Registered<ArrowFragmentGroup>, GlobalObject {
 public:
  static std::unique_ptr<vineyard::Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<ArrowFragmentGroup>{new ArrowFragmentGroup()});
  }
  fid_t total_frag_num() const { return total_frag_num_; }
  property_graph_types::LABEL_ID_TYPE vertex_label_num() const {
    return vertex_label_num_;
  }
  property_graph_types::LABEL_ID_TYPE edge_label_num() const {
    return edge_label_num_;
  }
  const std::unordered_map<fid_t, vineyard::ObjectID>& Fragments() {
    return fragments_;
  }
  const std::unordered_map<fid_t, uint64_t>& FragmentLocations() {
    return fragment_locations_;
  }

  void Construct(const vineyard::ObjectMeta& meta) override;

 private:
  fid_t total_frag_num_;
  property_graph_types::LABEL_ID_TYPE vertex_label_num_;
  property_graph_types::LABEL_ID_TYPE edge_label_num_;
  std::unordered_map<fid_t, vineyard::ObjectID> fragments_;
  std::unordered_map<fid_t, uint64_t> fragment_locations_;

  friend ArrowFragmentGroupBuilder;
};

class ArrowFragmentGroupBuilder : public ObjectBuilder {
 public:
  ArrowFragmentGroupBuilder() {}

  void set_total_frag_num(fid_t total_frag_num) {
    total_frag_num_ = total_frag_num;
  }
  void set_vertex_label_num(
      property_graph_types::LABEL_ID_TYPE vertex_label_num) {
    vertex_label_num_ = vertex_label_num;
  }
  void set_edge_label_num(property_graph_types::LABEL_ID_TYPE edge_label_num) {
    edge_label_num_ = edge_label_num;
  }
  void AddFragmentObject(fid_t fid, vineyard::ObjectID object_id,
                         uint64_t instance_id);

  vineyard::Status Build(vineyard::Client& client) override;

  Status _Seal(vineyard::Client& client,
               std::shared_ptr<vineyard::Object>& object) override;

 private:
  fid_t total_frag_num_;
  property_graph_types::LABEL_ID_TYPE vertex_label_num_;
  property_graph_types::LABEL_ID_TYPE edge_label_num_;
  std::unordered_map<fid_t, ObjectID> fragments_;
  std::unordered_map<fid_t, uint64_t> fragment_locations_;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GROUP_H_
