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

#ifndef MODULES_GRAPH_VERTEX_MAP_ARROW_LOCAL_VERTEX_MAP_IMPL_H_
#define MODULES_GRAPH_VERTEX_MAP_ARROW_LOCAL_VERTEX_MAP_IMPL_H_

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "basic/ds/array.h"
#include "basic/ds/arrow.h"
#include "basic/ds/hashmap.h"
#include "basic/utils.h"
#include "client/client.h"
#include "common/util/functions.h"
#include "common/util/typename.h"

#include "graph/fragment/property_graph_types.h"
#include "graph/fragment/property_graph_utils.h"
#include "graph/utils/thread_group.h"
#include "graph/vertex_map/arrow_local_vertex_map.h"

namespace vineyard {

template <typename OID_T, typename VID_T>
void ArrowLocalVertexMap<OID_T, VID_T>::Construct(
    const vineyard::ObjectMeta& meta) {
  this->meta_ = meta;
  this->id_ = meta.GetId();

  this->fnum_ = meta.GetKeyValue<fid_t>("fnum");
  this->fid_ = meta.GetKeyValue<fid_t>("fid");
  this->label_num_ = meta.GetKeyValue<label_id_t>("label_num");

  id_parser_.Init(fnum_, label_num_);

  size_t nbytes = 0, local_oid_total = 0;
  size_t o2i_total_bytes = 0, i2o_total_bytes = 0, o2i_size = 0,
         o2i_bucket_count = 0, i2o_size = 0, i2o_bucket_count = 0;

  oid_arrays_.resize(fnum_);
  o2i_.resize(fnum_);
  i2o_.resize(fnum_);
  i2o_index_.resize(fnum_);
  vertices_num_.resize(fnum_);
  for (fid_t fid = 0; fid < fnum_; ++fid) {
    oid_arrays_[fid].resize(label_num_);
    o2i_[fid].resize(label_num_);
    i2o_[fid].resize(label_num_);
    i2o_index_[fid].resize(label_num_);
    vertices_num_[fid].resize(label_num_);
    for (label_id_t label = 0; label < label_num_; ++label) {
      std::string suffix = std::to_string(fid) + "_" + std::to_string(label);

      typename InternalType<oid_t>::vineyard_array_type array;
      array.Construct(meta.GetMemberMeta("oid_arrays_" + suffix));
      oid_arrays_[fid][label] = array.GetArray();
      local_oid_total += array.nbytes();

      if (fid != fid_) {
        i2o_[fid][label].Construct(meta.GetMemberMeta("i2o_" + suffix));
        i2o_size += i2o_[fid][label].size();
        i2o_total_bytes += i2o_[fid][label].nbytes();
        i2o_bucket_count += i2o_[fid][label].bucket_count();

        i2o_index_[fid][label].Construct(
            meta.GetMemberMeta("i2o_index_" + suffix));
        i2o_size += i2o_index_[fid][label].size();
        i2o_total_bytes += i2o_index_[fid][label].nbytes();
        i2o_bucket_count += i2o_index_[fid][label].bucket_count();
      }

      o2i_[fid][label].Construct(meta.GetMemberMeta("o2i_" + suffix));
      o2i_size += o2i_[fid][label].size();
      o2i_total_bytes += o2i_[fid][label].nbytes();
      o2i_bucket_count += o2i_[fid][label].bucket_count();

      vertices_num_[fid][label] =
          meta.GetKeyValue<vid_t>("vertices_num_" + suffix);
    }
  }
  nbytes = local_oid_total + o2i_total_bytes + i2o_total_bytes;
  double o2i_load_factor =
      o2i_bucket_count == 0 ? 0
                            : static_cast<size_t>(o2i_size) / o2i_bucket_count;
  double i2o_load_factor =
      i2o_bucket_count == 0 ? 0
                            : static_cast<size_t>(i2o_size) / i2o_bucket_count;
  VLOG(100) << type_name<ArrowLocalVertexMap<oid_t, vid_t>>()
            << "\n\tmemory: " << prettyprint_memory_size(nbytes)
            << "\n\to2i size: " << o2i_size
            << ", load factor: " << o2i_load_factor
            << "\n\to2i memory: " << prettyprint_memory_size(o2i_total_bytes)
            << "\n\ti2o size: " << i2o_size
            << ", load factor: " << i2o_load_factor
            << "\n\ti2o memory: " << prettyprint_memory_size(i2o_total_bytes);
}

template <typename OID_T, typename VID_T>
bool ArrowLocalVertexMap<OID_T, VID_T>::GetOid(vid_t gid, oid_t& oid) const {
  fid_t fid = id_parser_.GetFid(gid);
  label_id_t label = id_parser_.GetLabelId(gid);
  int64_t offset = id_parser_.GetOffset(gid);
  if (fid < fnum_ && label < label_num_ && label >= 0) {
    if (fid == fid_) {
      if (offset < oid_arrays_[fid][label]->length()) {
        oid = oid_arrays_[fid][label]->GetView(offset);
        return true;
      }
    } else {
      if (!std::is_same<OID_T, arrow_string_view>::value) {
        // non string_view oid
        auto iter = i2o_[fid][label].find(offset);
        if (iter != i2o_[fid][label].end()) {
          oid = iter->second;
          return true;
        }
      } else {
        // string_view oid
        auto iter = i2o_index_[fid][label].find(offset);
        if (iter != i2o_index_[fid][label].end()) {
          oid = oid_arrays_[fid][label]->GetView(iter->second);
          return true;
        }
      }
    }
  }
  return false;
}

template <typename OID_T, typename VID_T>
bool ArrowLocalVertexMap<OID_T, VID_T>::GetGid(fid_t fid, label_id_t label_id,
                                               oid_t oid, vid_t& gid) const {
  auto iter = o2i_[fid][label_id].find(oid);
  if (iter != o2i_[fid][label_id].end()) {
    gid = id_parser_.GenerateId(fid, label_id, iter->second);
    return true;
  }
  return false;
}

template <typename OID_T, typename VID_T>
bool ArrowLocalVertexMap<OID_T, VID_T>::GetGid(label_id_t label_id, oid_t oid,
                                               vid_t& gid) const {
  for (fid_t i = 0; i < fnum_; ++i) {
    if (GetGid(i, label_id, oid, gid)) {
      return true;
    }
  }
  return false;
}

template <typename OID_T, typename VID_T>
std::vector<OID_T> ArrowLocalVertexMap<OID_T, VID_T>::GetOids(
    fid_t fid, label_id_t label_id) const {
  CHECK(fid == fid_);
  auto array = oid_arrays_[fid][label_id];
  std::vector<oid_t> oids;

  oids.resize(array->length());
  for (auto i = 0; i < array->length(); i++) {
    oids[i] = array->GetView(i);
  }

  return oids;
}

template <typename OID_T, typename VID_T>
std::shared_ptr<ArrowArrayType<OID_T>>
ArrowLocalVertexMap<OID_T, VID_T>::GetOidArray(fid_t fid,
                                               label_id_t label_id) const {
  CHECK(fid == fid_);
  return oid_arrays_[fid][label_id];
}

template <typename OID_T, typename VID_T>
size_t ArrowLocalVertexMap<OID_T, VID_T>::GetTotalNodesNum() const {
  size_t num = 0;
  for (auto& vec : vertices_num_) {
    for (auto& v : vec) {
      num += v;
    }
  }
  return num;
}

template <typename OID_T, typename VID_T>
size_t ArrowLocalVertexMap<OID_T, VID_T>::GetTotalNodesNum(
    label_id_t label) const {
  size_t num = 0;
  for (auto& vec : vertices_num_) {
    num += vec[label];
  }
  return num;
}

template <typename OID_T, typename VID_T>
VID_T ArrowLocalVertexMap<OID_T, VID_T>::GetInnerVertexSize(fid_t fid) const {
  size_t num = 0;
  for (auto& v : vertices_num_[fid]) {
    num += v;
  }
  return static_cast<vid_t>(num);
}

template <typename OID_T, typename VID_T>
VID_T ArrowLocalVertexMap<OID_T, VID_T>::GetInnerVertexSize(
    fid_t fid, label_id_t label_id) const {
  return static_cast<vid_t>(vertices_num_[fid][label_id]);
}

template <typename OID_T, typename VID_T>
ObjectID ArrowLocalVertexMap<OID_T, VID_T>::AddVertices(
    Client& client,
    const std::map<label_id_t, std::vector<std::shared_ptr<oid_array_t>>>&
        oid_arrays_map) {
  LOG(ERROR) << "ArrowLocalVertexMap not support AddVertices operation yet";
  return InvalidObjectID();
}

template <typename OID_T, typename VID_T>
ObjectID ArrowLocalVertexMap<OID_T, VID_T>::AddVertices(
    Client& client,
    const std::map<label_id_t,
                   std::vector<std::shared_ptr<arrow::ChunkedArray>>>&
        oid_arrays_map) {
  LOG(ERROR) << "ArrowLocalVertexMap not support AddVertices operation yet";
  return InvalidObjectID();
}

template <typename OID_T, typename VID_T>
ObjectID ArrowLocalVertexMap<OID_T, VID_T>::AddNewVertexLabels(
    Client& client,
    const std::vector<std::vector<std::shared_ptr<oid_array_t>>>& oid_arrays) {
  LOG(ERROR)
      << "ArrowLocalVertexMap not support AddNewVertexLabels operation yet";
  return InvalidObjectID();
}

template <typename OID_T, typename VID_T>
ObjectID ArrowLocalVertexMap<OID_T, VID_T>::AddNewVertexLabels(
    Client& client,
    const std::vector<std::vector<std::shared_ptr<arrow::ChunkedArray>>>&
        oid_arrays) {
  LOG(ERROR)
      << "ArrowLocalVertexMap not support AddNewVertexLabels operation yet";
  return InvalidObjectID();
}

template <typename OID_T, typename VID_T>
ArrowLocalVertexMapBuilder<OID_T, VID_T>::ArrowLocalVertexMapBuilder(
    vineyard::Client& client, fid_t fnum, fid_t fid, label_id_t label_num)
    : client(client), fnum_(fnum), fid_(fid), label_num_(label_num) {
  oid_arrays_.resize(fnum);
  o2i_.resize(fnum);
  i2o_.resize(fnum);
  i2o_index_.resize(fnum);
  for (fid_t fid = 0; fid < fnum_; ++fid) {
    oid_arrays_[fid].resize(label_num_);
    o2i_[fid].resize(label_num_);
    if (fid != fid_) {
      i2o_[fid].resize(label_num_);
      i2o_index_[fid].resize(label_num_);
    }
  }
  vertices_num_.resize(fnum_);
  for (fid_t fid = 0; fid < fnum_; ++fid) {
    vertices_num_[fid].resize(label_num_);
  }

  id_parser_.Init(fnum_, label_num_);
}

template <typename OID_T, typename VID_T>
vineyard::Status ArrowLocalVertexMapBuilder<OID_T, VID_T>::Build(
    vineyard::Client& client) {
  DLOG(WARNING) << "Empty 'Build' method.";
  return vineyard::Status::OK();
}

template <typename OID_T, typename VID_T>
Status ArrowLocalVertexMapBuilder<OID_T, VID_T>::_Seal(
    vineyard::Client& client, std::shared_ptr<vineyard::Object>& object) {
  // ensure the builder hasn't been sealed yet.
  ENSURE_NOT_SEALED(this);

  // empty method
  // VINEYARD_CHECK_OK(this->Build(client));

  auto vertex_map = std::make_shared<ArrowLocalVertexMap<oid_t, vid_t>>();
  object = vertex_map;

  vertex_map->fnum_ = fnum_;
  vertex_map->label_num_ = label_num_;
  vertex_map->id_parser_.Init(fnum_, label_num_);

  vertex_map->oid_arrays_.resize(fnum_);
  for (fid_t fid = 0; fid < fnum_; ++fid) {
    auto& arrays = vertex_map->oid_arrays_[fid];
    arrays.resize(label_num_);
    for (label_id_t label = 0; label < label_num_; ++label) {
      arrays[label] = oid_arrays_[fid][label].GetArray();
    }
  }

  vertex_map->o2i_ = o2i_;
  vertex_map->i2o_ = i2o_;
  vertex_map->i2o_index_ = i2o_index_;
  vertex_map->vertices_num_ = vertices_num_;

  vertex_map->meta_.SetTypeName(type_name<ArrowLocalVertexMap<oid_t, vid_t>>());

  vertex_map->meta_.AddKeyValue("fnum", fnum_);
  vertex_map->meta_.AddKeyValue("fid", fid_);
  vertex_map->meta_.AddKeyValue("label_num", label_num_);

  size_t nbytes = 0;
  for (fid_t fid = 0; fid < fnum_; ++fid) {
    for (label_id_t label = 0; label < label_num_; ++label) {
      std::string suffix = std::to_string(fid) + "_" + std::to_string(label);

      vertex_map->meta_.AddMember("oid_arrays_" + suffix,
                                  oid_arrays_[fid][label].meta());
      nbytes += oid_arrays_[fid][label].nbytes();

      vertex_map->meta_.AddMember("o2i_" + suffix, o2i_[fid][label].meta());
      nbytes += o2i_[fid][label].nbytes();
      if (fid != fid_) {
        vertex_map->meta_.AddMember("i2o_" + suffix, i2o_[fid][label].meta());
        nbytes += i2o_[fid][label].nbytes();
        vertex_map->meta_.AddMember("i2o_index_" + suffix,
                                    i2o_index_[fid][label].meta());
        nbytes += i2o_index_[fid][label].nbytes();
      }
      vertex_map->meta_.AddKeyValue("vertices_num_" + suffix,
                                    vertices_num_[fid][label]);
    }
  }

  vertex_map->meta_.SetNBytes(nbytes);

  RETURN_ON_ERROR(client.CreateMetaData(vertex_map->meta_, vertex_map->id_));
  // mark the builder as sealed
  this->set_sealed(true);
  return Status::OK();
}

template <typename OID_T, typename VID_T>
vineyard::Status ArrowLocalVertexMapBuilder<OID_T, VID_T>::AddLocalVertices(
    grape::CommSpec& comm_spec,
    std::vector<std::shared_ptr<oid_array_t>> oid_arrays) {
  std::vector<std::vector<std::shared_ptr<oid_array_t>>> arrays(
      oid_arrays.size());
  for (size_t i = 0; i < oid_arrays.size(); ++i) {
    arrays[i] = {oid_arrays[i]};
  }
  return addLocalVertices(comm_spec, std::move(arrays));
}

template <typename OID_T, typename VID_T>
vineyard::Status ArrowLocalVertexMapBuilder<OID_T, VID_T>::AddLocalVertices(
    grape::CommSpec& comm_spec,
    std::vector<std::shared_ptr<arrow::ChunkedArray>> oid_arrays) {
  std::vector<std::vector<std::shared_ptr<oid_array_t>>> arrays(
      oid_arrays.size());
  for (size_t i = 0; i < oid_arrays.size(); ++i) {
    for (auto const& chunk : oid_arrays[i]->chunks()) {
      arrays[i].emplace_back(std::dynamic_pointer_cast<oid_array_t>(chunk));
    }
  }
  return addLocalVertices(comm_spec, std::move(arrays));
}

template <typename OID_T, typename VID_T>
vineyard::Status ArrowLocalVertexMapBuilder<OID_T, VID_T>::addLocalVertices(
    grape::CommSpec& comm_spec,
    std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_arrays) {
  auto fn = [&](label_id_t label) -> Status {
    auto& arrays = oid_arrays[label];
    typename InternalType<oid_t>::vineyard_builder_type array_builder(client,
                                                                      arrays);
    std::shared_ptr<Object> object;
    RETURN_ON_ERROR(array_builder.Seal(client, object));
    oid_arrays_[fid_][label] = *std::dynamic_pointer_cast<
        typename InternalType<oid_t>::vineyard_array_type>(object);

    // release the reference
    arrays.clear();

    // now use the array in vineyard memory
    auto array = oid_arrays_[fid_][label].GetArray();
    vineyard::HashmapBuilder<oid_t, vid_t> builder(client);
    int64_t vnum = array->length();
    builder.reserve(static_cast<size_t>(vnum));
    for (int64_t i = 0; i < vnum; ++i) {
      if (!builder.emplace(array->GetView(i), i)) {
        LOG(WARNING)
            << "The vertex '" << array->GetView(i) << "' has been added "
            << "more than once, please double check your vertices data";
      }
    }
    RETURN_ON_ERROR(builder.Seal(client, object));
    o2i_[fid_][label] =
        *std::dynamic_pointer_cast<vineyard::Hashmap<oid_t, vid_t>>(object);

    vertices_num_[fid_][label] = vnum;
    return Status::OK();
  };

  ThreadGroup tg(comm_spec);
  for (label_id_t label = 0; label < label_num_; ++label) {
    tg.AddTask(fn, label);
  }

  Status status;
  for (auto const& s : tg.TakeResults()) {
    status += s;
  }
  RETURN_ON_ERROR(status);

  // sync the vertices_num
  for (label_id_t label = 0; label < label_num_; ++label) {
    std::vector<vid_t> current_vertices_num(fnum_);
    current_vertices_num[fid_] = vertices_num_[fid_][label];
    grape::sync_comm::AllGather(current_vertices_num, comm_spec.comm());
    for (fid_t fid = 0; fid < fnum_; ++fid) {
      vertices_num_[fid][label] = current_vertices_num[fid];
    }
  }
  return vineyard::Status::OK();
}

template <typename OID_T, typename VID_T>
vineyard::Status ArrowLocalVertexMapBuilder<OID_T, VID_T>::GetIndexOfOids(
    const std::vector<std::shared_ptr<oid_array_t>>& oids,
    std::vector<std::vector<vid_t>>& index_list) {
  index_list.resize(label_num_);

  for (label_id_t label_id = 0; label_id < label_num_; ++label_id) {
    auto& o2i_map = o2i_[fid_][label_id];
    auto& current_oids = oids[label_id];
    auto& current_index_list = index_list[label_id];
    current_index_list.resize(current_oids->length());

    parallel_for(
        static_cast<int64_t>(0), current_oids->length(),
        [&](const size_t& i) {
          current_index_list[i] =
              o2i_map.find(current_oids->GetView(i))->second;
        },
        std::thread::hardware_concurrency());
  }
  return vineyard::Status::OK();
}

template <typename OID_T, typename VID_T>
template <typename OID_TYPE, typename std::enable_if<!std::is_same<
                                 OID_TYPE, arrow_string_view>::value>::type*>
vineyard::Status
ArrowLocalVertexMapBuilder<OID_T, VID_T>::AddOuterVerticesMapping(
    std::vector<std::vector<std::shared_ptr<ArrowArrayType<OID_TYPE>>>> oids,
    std::vector<std::vector<std::vector<vid_t>>> index_list) {
  auto fn = [&](fid_t cur_fid, label_id_t cur_label) -> Status {
    // filling an empty oid array for non-local fragments
    std::shared_ptr<ArrowArrayType<oid_t>> oid_array;
    ArrowBuilderType<oid_t> array_builder;
    RETURN_ON_ARROW_ERROR(array_builder.Finish(&oid_array));
    typename InternalType<oid_t>::vineyard_builder_type outer_oid_builder(
        client, oid_array);
    std::shared_ptr<Object> object;
    RETURN_ON_ERROR(outer_oid_builder.Seal(client, object));
    oid_arrays_[cur_fid][cur_label] = *std::dynamic_pointer_cast<
        typename InternalType<oid_t>::vineyard_array_type>(object);

    std::shared_ptr<oid_array_t>& current_oid_array = oids[cur_fid][cur_label];

    vineyard::HashmapBuilder<oid_t, vid_t> o2i_builder(client);
    vineyard::HashmapBuilder<vid_t, oid_t> i2o_builder(client);
    vineyard::HashmapBuilder<vid_t, vid_t> i2o_index_builder(client);
    o2i_builder.reserve(static_cast<size_t>(current_oid_array->length()));
    i2o_builder.reserve(static_cast<size_t>(current_oid_array->length()));
    for (int64_t i = 0; i < current_oid_array->length(); i++) {
      auto oid = current_oid_array->GetView(i);
      auto& index = index_list[cur_fid][cur_label][i];
      o2i_builder.emplace(oid, index);
      i2o_builder.emplace(index, oid);
    }

    // release the reference
    oids[cur_fid][cur_label].reset();
    index_list[cur_fid][cur_label].clear();
    index_list[cur_fid][cur_label].shrink_to_fit();

    RETURN_ON_ERROR(o2i_builder.Seal(client, object));
    o2i_[cur_fid][cur_label] =
        *std::dynamic_pointer_cast<vineyard::Hashmap<oid_t, vid_t>>(object);
    RETURN_ON_ERROR(i2o_builder.Seal(client, object));
    i2o_[cur_fid][cur_label] =
        *std::dynamic_pointer_cast<vineyard::Hashmap<vid_t, oid_t>>(object);
    RETURN_ON_ERROR(i2o_index_builder.Seal(client, object));
    i2o_index_[cur_fid][cur_label] =
        *std::dynamic_pointer_cast<vineyard::Hashmap<vid_t, vid_t>>(object);
    return Status::OK();
  };

  ThreadGroup tg;
  for (fid_t fid = 0; fid < fnum_; ++fid) {
    if (fid == fid_) {
      continue;
    }
    for (label_id_t label = 0; label < label_num_; ++label) {
      tg.AddTask(fn, fid, label);
    }
  }
  Status status;
  for (auto const& s : tg.TakeResults()) {
    status += s;
  }
  return status;
}

template <typename OID_T, typename VID_T>
template <typename OID_TYPE, typename std::enable_if<std::is_same<
                                 OID_TYPE, arrow_string_view>::value>::type*>
vineyard::Status
ArrowLocalVertexMapBuilder<OID_T, VID_T>::AddOuterVerticesMapping(
    std::vector<std::vector<std::shared_ptr<ArrowArrayType<OID_TYPE>>>> oids,
    std::vector<std::vector<std::vector<vid_t>>> index_list) {
  auto fn = [&](fid_t cur_fid, label_id_t cur_label) -> Status {
    typename InternalType<oid_t>::vineyard_builder_type outer_oid_builder(
        client, oids[cur_fid][cur_label]);
    std::shared_ptr<Object> object;
    RETURN_ON_ERROR(outer_oid_builder.Seal(client, object));
    oid_arrays_[cur_fid][cur_label] = *std::dynamic_pointer_cast<
        typename InternalType<oid_t>::vineyard_array_type>(object);

    // release the reference
    oids[cur_fid][cur_label].reset();

    std::shared_ptr<oid_array_t> current_oid_array =
        oid_arrays_[cur_fid][cur_label].GetArray();

    vineyard::HashmapBuilder<oid_t, vid_t> o2i_builder(client);
    vineyard::HashmapBuilder<vid_t, oid_t> i2o_builder(client);
    vineyard::HashmapBuilder<vid_t, vid_t> i2o_index_builder(client);
    o2i_builder.reserve(static_cast<size_t>(current_oid_array->length()));
    // n.b.
    o2i_builder.AssociateDataBuffer(
        oid_arrays_[cur_fid][cur_label].GetBuffer());
    i2o_index_builder.reserve(static_cast<size_t>(current_oid_array->length()));
    for (int64_t i = 0; i < current_oid_array->length(); i++) {
      auto oid = current_oid_array->GetView(i);
      auto& index = index_list[cur_fid][cur_label][i];
      o2i_builder.emplace(oid, index);
      i2o_index_builder.emplace(index, static_cast<size_t>(i));
    }

    // release the reference
    index_list[cur_fid][cur_label].clear();
    index_list[cur_fid][cur_label].shrink_to_fit();

    RETURN_ON_ERROR(o2i_builder.Seal(client, object));
    o2i_[cur_fid][cur_label] =
        *std::dynamic_pointer_cast<vineyard::Hashmap<oid_t, vid_t>>(object);
    RETURN_ON_ERROR(i2o_builder.Seal(client, object));
    i2o_[cur_fid][cur_label] =
        *std::dynamic_pointer_cast<vineyard::Hashmap<vid_t, oid_t>>(object);
    RETURN_ON_ERROR(i2o_index_builder.Seal(client, object));
    i2o_index_[cur_fid][cur_label] =
        *std::dynamic_pointer_cast<vineyard::Hashmap<vid_t, vid_t>>(object);
    return Status::OK();
  };

  ThreadGroup tg;
  for (fid_t fid = 0; fid < fnum_; ++fid) {
    if (fid == fid_) {
      continue;
    }
    for (label_id_t label = 0; label < label_num_; ++label) {
      tg.AddTask(fn, fid, label);
    }
  }
  Status status;
  for (auto const& s : tg.TakeResults()) {
    status += s;
  }
  return status;
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_VERTEX_MAP_ARROW_LOCAL_VERTEX_MAP_IMPL_H_
