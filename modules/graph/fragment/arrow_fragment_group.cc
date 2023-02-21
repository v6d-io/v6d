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

#include "graph/fragment/arrow_fragment_group.h"

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "client/client.h"

#include "graph/fragment/property_graph_types.h"

namespace vineyard {

void ArrowFragmentGroup::Construct(const vineyard::ObjectMeta& meta) {
  this->meta_ = meta;
  this->id_ = meta.GetId();

  total_frag_num_ = meta.GetKeyValue<fid_t>("total_frag_num");
  vertex_label_num_ =
      meta.GetKeyValue<property_graph_types::LABEL_ID_TYPE>("vertex_label_num");
  edge_label_num_ =
      meta.GetKeyValue<property_graph_types::LABEL_ID_TYPE>("edge_label_num");
  for (fid_t idx = 0; idx < total_frag_num_; ++idx) {
    fragments_.emplace(
        meta.GetKeyValue<fid_t>("fid_" + std::to_string(idx)),
        meta.GetMemberMeta("frag_object_id_" + std::to_string(idx)).GetId());
    fragment_locations_.emplace(
        meta.GetKeyValue<fid_t>("fid_" + std::to_string(idx)),
        meta.GetKeyValue<uint64_t>("frag_instance_id_" + std::to_string(idx)));
  }
}

void ArrowFragmentGroupBuilder::AddFragmentObject(fid_t fid,
                                                  vineyard::ObjectID object_id,
                                                  uint64_t instance_id) {
  fragments_.emplace(fid, object_id);
  fragment_locations_.emplace(fid, instance_id);
}

vineyard::Status ArrowFragmentGroupBuilder::Build(vineyard::Client& client) {
  return vineyard::Status::OK();
}

Status ArrowFragmentGroupBuilder::_Seal(
    vineyard::Client& client, std::shared_ptr<vineyard::Object>& object) {
  // ensure the builder hasn't been sealed yet.
  ENSURE_NOT_SEALED(this);

  RETURN_ON_ERROR(this->Build(client));

  auto fg = std::make_shared<ArrowFragmentGroup>();
  object = fg;

  fg->total_frag_num_ = total_frag_num_;
  fg->vertex_label_num_ = vertex_label_num_;
  fg->edge_label_num_ = edge_label_num_;
  fg->fragments_ = fragments_;
  if (std::is_base_of<GlobalObject, ArrowFragmentGroup>::value) {
    fg->meta_.SetGlobal(true);
  }
  fg->meta_.SetTypeName(type_name<ArrowFragmentGroup>());
  fg->meta_.AddKeyValue("total_frag_num", total_frag_num_);
  fg->meta_.AddKeyValue("vertex_label_num", vertex_label_num_);
  fg->meta_.AddKeyValue("edge_label_num", edge_label_num_);
  int idx = 0;

  for (auto const& kv : fragments_) {
    fg->meta_.AddKeyValue("fid_" + std::to_string(idx), kv.first);
    fg->meta_.AddKeyValue("frag_instance_id_" + std::to_string(idx),
                          fragment_locations_[kv.first]);
    fg->meta_.AddMember("frag_object_id_" + std::to_string(idx), kv.second);
    idx += 1;
  }

  RETURN_ON_ERROR(client.CreateMetaData(fg->meta_, fg->id_));
  // mark the builder as sealed
  this->set_sealed(true);
  return Status::OK();
}

}  // namespace vineyard
