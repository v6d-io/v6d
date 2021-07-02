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

#ifndef MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GROUP_H_
#define MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GROUP_H_

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "client/client.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/property_graph_types.h"

namespace vineyard {

class ArrowFragmentGroupBuilder;

class ArrowFragmentGroup : public Registered<ArrowFragmentGroup>, GlobalObject {
 public:
  static std::shared_ptr<vineyard::Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::make_shared<ArrowFragmentGroup>());
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

  void Construct(const vineyard::ObjectMeta& meta) override {
    this->meta_ = meta;
    this->id_ = meta.GetId();

    total_frag_num_ = meta.GetKeyValue<fid_t>("total_frag_num");
    vertex_label_num_ = meta.GetKeyValue<property_graph_types::LABEL_ID_TYPE>(
        "vertex_label_num");
    edge_label_num_ =
        meta.GetKeyValue<property_graph_types::LABEL_ID_TYPE>("edge_label_num");
    for (fid_t idx = 0; idx < total_frag_num_; ++idx) {
      fragments_.emplace(
          meta.GetKeyValue<fid_t>("fid_" + std::to_string(idx)),
          meta.GetMemberMeta("frag_object_id_" + std::to_string(idx)).GetId());
      fragment_locations_.emplace(
          meta.GetKeyValue<fid_t>("fid_" + std::to_string(idx)),
          meta.GetKeyValue<uint64_t>("frag_instance_id_" +
                                     std::to_string(idx)));
    }
  }

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
                         uint64_t instance_id) {
    fragments_.emplace(fid, object_id);
    fragment_locations_.emplace(fid, instance_id);
  }

  vineyard::Status Build(vineyard::Client& client) override {
    return vineyard::Status::OK();
  }

  std::shared_ptr<vineyard::Object> _Seal(vineyard::Client& client) override {
    // ensure the builder hasn't been sealed yet.
    ENSURE_NOT_SEALED(this);

    VINEYARD_CHECK_OK(this->Build(client));

    auto fg = std::make_shared<ArrowFragmentGroup>();
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

    VINEYARD_CHECK_OK(client.CreateMetaData(fg->meta_, fg->id_));
    // mark the builder as sealed
    this->set_sealed(true);

    return std::static_pointer_cast<vineyard::Object>(fg);
  }

 private:
  fid_t total_frag_num_;
  property_graph_types::LABEL_ID_TYPE vertex_label_num_;
  property_graph_types::LABEL_ID_TYPE edge_label_num_;
  std::unordered_map<fid_t, ObjectID> fragments_;
  std::unordered_map<fid_t, uint64_t> fragment_locations_;
};

inline boost::leaf::result<ObjectID> ConstructFragmentGroup(
    Client& client, ObjectID frag_id, const grape::CommSpec& comm_spec) {
  ObjectID group_object_id;
  uint64_t instance_id = client.instance_id();

  MPI_Barrier(comm_spec.comm());
  VINEYARD_DISCARD(client.SyncMetaData());

  if (comm_spec.worker_id() == 0) {
    std::vector<uint64_t> gathered_instance_ids(comm_spec.worker_num());
    std::vector<ObjectID> gathered_object_ids(comm_spec.worker_num());

    MPI_Gather(&instance_id, sizeof(uint64_t), MPI_CHAR,
               &gathered_instance_ids[0], sizeof(uint64_t), MPI_CHAR, 0,
               comm_spec.comm());

    MPI_Gather(&frag_id, sizeof(ObjectID), MPI_CHAR, &gathered_object_ids[0],
               sizeof(ObjectID), MPI_CHAR, 0, comm_spec.comm());

    ArrowFragmentGroupBuilder builder;
    builder.set_total_frag_num(comm_spec.fnum());
    auto fragment =
        std::dynamic_pointer_cast<ArrowFragmentBase>(client.GetObject(frag_id));
    auto& meta = fragment->meta();

    builder.set_vertex_label_num(
        meta.GetKeyValue<typename ArrowFragmentBase::label_id_t>(
            "vertex_label_num"));
    builder.set_edge_label_num(
        meta.GetKeyValue<typename ArrowFragmentBase::label_id_t>(
            "edge_label_num"));
    for (fid_t i = 0; i < comm_spec.fnum(); ++i) {
      builder.AddFragmentObject(
          i, gathered_object_ids[comm_spec.FragToWorker(i)],
          gathered_instance_ids[comm_spec.FragToWorker(i)]);
    }

    auto group_object =
        std::dynamic_pointer_cast<ArrowFragmentGroup>(builder.Seal(client));
    group_object_id = group_object->id();
    VY_OK_OR_RAISE(client.Persist(group_object_id));

    MPI_Bcast(&group_object_id, sizeof(ObjectID), MPI_CHAR, 0,
              comm_spec.comm());
  } else {
    MPI_Gather(&instance_id, sizeof(uint64_t), MPI_CHAR, NULL, sizeof(uint64_t),
               MPI_CHAR, 0, comm_spec.comm());
    MPI_Gather(&frag_id, sizeof(ObjectID), MPI_CHAR, NULL, sizeof(ObjectID),
               MPI_CHAR, 0, comm_spec.comm());

    MPI_Bcast(&group_object_id, sizeof(ObjectID), MPI_CHAR, 0,
              comm_spec.comm());
  }

  MPI_Barrier(comm_spec.comm());
  VINEYARD_DISCARD(client.SyncMetaData());
  return group_object_id;
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_FRAGMENT_ARROW_FRAGMENT_GROUP_H_
