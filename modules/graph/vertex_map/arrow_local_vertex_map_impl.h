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
  local_oid_arrays_.resize(label_num_);
  for (label_id_t i = 0; i < label_num_; ++i) {
    typename InternalType<oid_t>::vineyard_array_type array;
    array.Construct(
        meta.GetMemberMeta("local_oid_arrays_" + std::to_string(i)));
    local_oid_arrays_[i] = array.GetArray();
    local_oid_total += array.nbytes();
  }

  o2i_.resize(fnum_);
  i2o_.resize(fnum_);
  vertices_num_.resize(fnum_);
  for (fid_t i = 0; i < fnum_; ++i) {
    o2i_[i].resize(label_num_);
    i2o_[i].resize(label_num_);
    vertices_num_[i].resize(label_num_);
    for (label_id_t j = 0; j < label_num_; ++j) {
      o2i_[i][j].Construct(meta.GetMemberMeta("o2i_" + std::to_string(i) + "_" +
                                              std::to_string(j)));
      if (i != fid_) {
        i2o_[i][j].Construct(meta.GetMemberMeta("i2o_" + std::to_string(i) +
                                                "_" + std::to_string(j)));
        i2o_size += i2o_[i][j].size();
        i2o_total_bytes += i2o_[i][j].nbytes();
        i2o_bucket_count += i2o_[i][j].bucket_count();
      }
      vertices_num_[i][j] = meta.GetKeyValue<vid_t>(
          "vertices_num_" + std::to_string(i) + "_" + std::to_string(j));

      o2i_size += o2i_[i][j].size();
      o2i_total_bytes += o2i_[i][j].nbytes();
      o2i_bucket_count += o2i_[i][j].bucket_count();
    }
  }
  nbytes = local_oid_total + o2i_total_bytes + i2o_total_bytes;
  double o2i_load_factor =
      o2i_bucket_count == 0 ? 0
                            : static_cast<size_t>(o2i_size) / o2i_bucket_count;
  double i2o_load_factor =
      i2o_bucket_count == 0 ? 0
                            : static_cast<size_t>(i2o_size) / i2o_bucket_count;
  VLOG(100) << "ArrowLocalVertexMap<int64_t, uint64_t> "
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
      if (offset < local_oid_arrays_[label]->length()) {
        oid = local_oid_arrays_[label]->GetView(offset);
        return true;
      }
    } else {
      auto iter = i2o_[fid][label].find(offset);
      if (iter != i2o_[fid][label].end()) {
        oid = iter->second;
        return true;
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
  auto array = local_oid_arrays_[label_id];
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
  return local_oid_arrays_[label_id];
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

template <typename VID_T>
void ArrowLocalVertexMap<arrow_string_view, VID_T>::Construct(
    const vineyard::ObjectMeta& meta) {
  this->meta_ = meta;
  this->id_ = meta.GetId();

  this->fnum_ = meta.GetKeyValue<fid_t>("fnum");
  this->fid_ = meta.GetKeyValue<fid_t>("fid");
  this->label_num_ = meta.GetKeyValue<label_id_t>("label_num");

  id_parser_.Init(fnum_, label_num_);

  oid_arrays_.resize(fnum_);
  index_arrays_.resize(fnum_);
  vertices_num_.resize(fnum_);
  size_t nbytes = 0, local_oid_total = 0, index_array_total = 0;
  size_t o2i_total_bytes = 0, i2o_total_bytes = 0, o2i_size = 0,
         o2i_bucket_count = 0, i2o_size = 0, i2o_bucket_count = 0;
  for (fid_t fid = 0; fid < fnum_; ++fid) {
    oid_arrays_[fid].resize(label_num_);
    if (fid != fid_) {
      index_arrays_[fid].resize(label_num_);
    }
    vertices_num_[fid].resize(label_num_);

    for (label_id_t label = 0; label < label_num_; ++label) {
      typename InternalType<oid_t>::vineyard_array_type array;
      array.Construct(meta.GetMemberMeta("oid_arrays_" + std::to_string(fid) +
                                         "_" + std::to_string(label)));
      oid_arrays_[fid][label] = array.GetArray();
      local_oid_total += array.nbytes();
      if (fid != fid_) {
        typename InternalType<vid_t>::vineyard_array_type index_array;
        index_array.Construct(meta.GetMemberMeta("index_arrays_" +
                                                 std::to_string(fid) + "_" +
                                                 std::to_string(label)));
        index_arrays_[fid][label] = index_array.GetArray();
        index_array_total += index_array.nbytes();
      }
      vertices_num_[fid][label] = meta.GetKeyValue<vid_t>(
          "vertices_num_" + std::to_string(fid) + "_" + std::to_string(label));
    }
  }

  initHashmaps();

  for (fid_t i = 0; i < fnum_; ++i) {
    for (label_id_t j = 0; j < label_num_; ++j) {
      if (i != fid_) {
        i2o_size += i2o_[i][j].size();
        i2o_bucket_count += i2o_[i][j].bucket_count();
        // 24 bytes = key(8) + value (8 (pointer) + 8 (length))
        i2o_total_bytes += i2o_[i][j].bucket_count() * 24;
      }
      o2i_size += o2i_[i][j].size();
      o2i_bucket_count += o2i_[i][j].bucket_count();
      // 24 bytes = key(8) + value (8 (pointer) + 8 (length))
      o2i_total_bytes += o2i_[i][j].bucket_count() * 24;
    }
  }

  nbytes =
      local_oid_total + index_array_total + i2o_total_bytes + o2i_total_bytes;
  double o2i_load_factor =
      o2i_bucket_count == 0 ? 0
                            : static_cast<double>(o2i_size) / o2i_bucket_count;
  double i2o_load_factor =
      i2o_bucket_count == 0 ? 0
                            : static_cast<double>(i2o_size) / i2o_bucket_count;
  VLOG(100) << "ArrowLocalVertexMap<string_view, uint64_t> "
            << "\n\tmemory: " << prettyprint_memory_size(nbytes)
            << "\n\tindex array: " << prettyprint_memory_size(index_array_total)
            << "\n\to2i size: " << o2i_size
            << ", load factor: " << o2i_load_factor
            << "\n\to2i memory: " << prettyprint_memory_size(o2i_total_bytes)
            << "\n\ti2o size: " << i2o_size
            << ", load factor: " << i2o_load_factor
            << "\n\ti2o memory: " << prettyprint_memory_size(i2o_total_bytes);
}

template <typename VID_T>
bool ArrowLocalVertexMap<arrow_string_view, VID_T>::GetOid(vid_t gid,
                                                           oid_t& oid) const {
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
      auto iter = i2o_[fid][label].find(offset);
      if (iter != i2o_[fid][label].end()) {
        oid = iter->second;
        return true;
      }
    }
  }
  return false;
}

template <typename VID_T>
bool ArrowLocalVertexMap<arrow_string_view, VID_T>::GetGid(fid_t fid,
                                                           label_id_t label_id,
                                                           oid_t oid,
                                                           vid_t& gid) const {
  auto iter = o2i_[fid][label_id].find(oid);
  if (iter != o2i_[fid][label_id].end()) {
    gid = id_parser_.GenerateId(fid, label_id, iter->second);
    return true;
  }
  return false;
}

template <typename VID_T>
bool ArrowLocalVertexMap<arrow_string_view, VID_T>::GetGid(label_id_t label_id,
                                                           oid_t oid,
                                                           vid_t& gid) const {
  for (fid_t i = 0; i < fnum_; ++i) {
    if (GetGid(i, label_id, oid, gid)) {
      return true;
    }
  }
  return false;
}

template <typename VID_T>
std::vector<arrow_string_view>
ArrowLocalVertexMap<arrow_string_view, VID_T>::GetOids(
    fid_t fid, label_id_t label_id) const {
  CHECK(fid == fid_);
  auto& array = oid_arrays_[fid][label_id];
  std::vector<oid_t> oids;

  oids.resize(array->length());
  for (auto i = 0; i < array->length(); i++) {
    oids[i] = array->GetView(i);
  }

  return oids;
}

template <typename VID_T>
size_t ArrowLocalVertexMap<arrow_string_view, VID_T>::GetTotalNodesNum() const {
  size_t num = 0;
  for (auto& vec : vertices_num_) {
    for (auto& v : vec) {
      num += v;
    }
  }
  return num;
}

template <typename VID_T>
size_t ArrowLocalVertexMap<arrow_string_view, VID_T>::GetTotalNodesNum(
    label_id_t label) const {
  size_t num = 0;
  for (auto& vec : vertices_num_) {
    num += vec[label];
  }
  return num;
}

template <typename VID_T>
VID_T ArrowLocalVertexMap<arrow_string_view, VID_T>::GetInnerVertexSize(
    fid_t fid) const {
  size_t num = 0;
  for (auto& v : vertices_num_[fid]) {
    num += v;
  }
  return static_cast<vid_t>(num);
}

template <typename VID_T>
VID_T ArrowLocalVertexMap<arrow_string_view, VID_T>::GetInnerVertexSize(
    fid_t fid, label_id_t label_id) const {
  return static_cast<vid_t>(vertices_num_[fid][label_id]);
}

template <typename VID_T>
ObjectID ArrowLocalVertexMap<arrow_string_view, VID_T>::AddVertices(
    Client& client,
    const std::map<label_id_t, std::vector<std::shared_ptr<oid_array_t>>>&
        oid_arrays_map) {
  LOG(ERROR) << "ArrowLocalVertexMap not support AddVertices operation yet";
  return InvalidObjectID();
}

template <typename VID_T>
ObjectID ArrowLocalVertexMap<arrow_string_view, VID_T>::AddVertices(
    Client& client,
    const std::map<label_id_t,
                   std::vector<std::shared_ptr<arrow::ChunkedArray>>>&
        oid_arrays_map) {
  LOG(ERROR) << "ArrowLocalVertexMap not support AddVertices operation yet";
  return InvalidObjectID();
}

template <typename VID_T>
ObjectID ArrowLocalVertexMap<arrow_string_view, VID_T>::AddNewVertexLabels(
    Client& client,
    const std::vector<std::vector<std::shared_ptr<oid_array_t>>>& oid_arrays) {
  LOG(ERROR)
      << "ArrowLocalVertexMap not support AddNewVertexLabels operation yet";
  return InvalidObjectID();
}

template <typename VID_T>
ObjectID ArrowLocalVertexMap<arrow_string_view, VID_T>::AddNewVertexLabels(
    Client& client,
    const std::vector<std::vector<std::shared_ptr<arrow::ChunkedArray>>>&
        oid_arrays) {
  LOG(ERROR)
      << "ArrowLocalVertexMap not support AddNewVertexLabels operation yet";
  return InvalidObjectID();
}

template <typename VID_T>
void ArrowLocalVertexMap<arrow_string_view, VID_T>::initHashmaps() {
  o2i_.resize(fnum_);
  i2o_.resize(fnum_);
  for (fid_t i = 0; i < fnum_; ++i) {
    o2i_[i].resize(label_num_);
    if (i != fid_) {
      i2o_[i].resize(label_num_);
    }
  }

  auto fn = [&](fid_t cur_fid, label_id_t cur_label) -> Status {
    int64_t vnum = oid_arrays_[cur_fid][cur_label]->length();
    if (cur_fid == fid_) {
      o2i_[cur_fid][cur_label].reserve(static_cast<size_t>(vnum));
      for (int64_t i = 0; i < vnum; ++i) {
        auto oid = oid_arrays_[cur_fid][cur_label]->GetView(i);
        o2i_[cur_fid][cur_label].emplace(oid, i);
      }
    } else {
      o2i_[cur_fid][cur_label].reserve(static_cast<size_t>(vnum));
      i2o_[cur_fid][cur_label].reserve(static_cast<size_t>(vnum));
      for (int64_t i = 0; i < vnum; ++i) {
        auto oid = oid_arrays_[cur_fid][cur_label]->GetView(i);
        auto index = index_arrays_[cur_fid][cur_label]->GetView(i);
        o2i_[cur_fid][cur_label].emplace(oid, index);
        i2o_[cur_fid][cur_label].emplace(index, oid);
      }
    }
    // shrink the size of hashmap
    o2i_[cur_fid][cur_label].shrink_to_fit();
    if (cur_fid != fid_) {
      i2o_[cur_fid][cur_label].shrink_to_fit();
    }
    return Status::OK();
  };

  ThreadGroup tg;
  for (fid_t fid = 0; fid < fnum_; ++fid) {
    for (label_id_t label = 0; label < label_num_; ++label) {
      tg.AddTask(fn, fid, label);
    }
  }
  Status status;
  for (auto const& s : tg.TakeResults()) {
    status += s;
  }
  VINEYARD_CHECK_OK(status);
}

template <typename OID_T, typename VID_T>
ArrowLocalVertexMapBuilder<OID_T, VID_T>::ArrowLocalVertexMapBuilder(
    vineyard::Client& client, fid_t fnum, fid_t fid, label_id_t label_num)
    : client(client), fnum_(fnum), fid_(fid), label_num_(label_num) {
  local_oid_arrays_.resize(label_num);
  o2i_.resize(fnum);
  o2i_[fid].resize(label_num);
  i2o_.resize(fnum);
  id_parser_.Init(fnum_, label_num_);
}

template <typename OID_T, typename VID_T>
vineyard::Status ArrowLocalVertexMapBuilder<OID_T, VID_T>::Build(
    vineyard::Client& client) {
  LOG(WARNING) << "Empty 'Build' method.";
  return vineyard::Status::OK();
}

template <typename OID_T, typename VID_T>
std::shared_ptr<vineyard::Object>
ArrowLocalVertexMapBuilder<OID_T, VID_T>::_Seal(vineyard::Client& client) {
  // ensure the builder hasn't been sealed yet.
  ENSURE_NOT_SEALED(this);

  auto vertex_map = std::make_shared<ArrowLocalVertexMap<oid_t, vid_t>>();
  vertex_map->fnum_ = fnum_;
  vertex_map->label_num_ = label_num_;
  vertex_map->id_parser_.Init(fnum_, label_num_);

  vertex_map->local_oid_arrays_.resize(label_num_);
  for (label_id_t i = 0; i < label_num_; ++i) {
    vertex_map->local_oid_arrays_[i] = local_oid_arrays_[i].GetArray();
  }

  vertex_map->o2i_ = o2i_;
  vertex_map->i2o_ = i2o_;

  vertex_map->meta_.SetTypeName(type_name<ArrowLocalVertexMap<oid_t, vid_t>>());

  vertex_map->meta_.AddKeyValue("fnum", fnum_);
  vertex_map->meta_.AddKeyValue("fid", fid_);
  vertex_map->meta_.AddKeyValue("label_num", label_num_);

  size_t nbytes = 0;
  for (label_id_t i = 0; i < label_num_; ++i) {
    vertex_map->meta_.AddMember("local_oid_arrays_" + std::to_string(i),
                                local_oid_arrays_[i].meta());
    nbytes += local_oid_arrays_[i].nbytes();
  }

  for (fid_t i = 0; i < fnum_; ++i) {
    for (label_id_t j = 0; j < label_num_; ++j) {
      vertex_map->meta_.AddKeyValue(
          "vertices_num_" + std::to_string(i) + "_" + std::to_string(j),
          vertices_num_[j][i]);
      vertex_map->meta_.AddMember(
          "o2i_" + std::to_string(i) + "_" + std::to_string(j),
          o2i_[i][j].meta());
      nbytes += o2i_[i][j].nbytes();
      if (i != fid_) {
        vertex_map->meta_.AddMember(
            "i2o_" + std::to_string(i) + "_" + std::to_string(j),
            i2o_[i][j].meta());
        nbytes += i2o_[i][j].nbytes();
      }
    }
  }

  vertex_map->meta_.SetNBytes(nbytes);

  VINEYARD_CHECK_OK(client.CreateMetaData(vertex_map->meta_, vertex_map->id_));
  // mark the builder as sealed
  this->set_sealed(true);

  return std::static_pointer_cast<vineyard::Object>(vertex_map);
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
  vertices_num_.resize(label_num_);

  auto fn = [&](label_id_t label) -> Status {
    auto& arrays = oid_arrays[label];
    typename InternalType<oid_t>::vineyard_builder_type array_builder(client,
                                                                      arrays);
    local_oid_arrays_[label] = *std::dynamic_pointer_cast<
        typename InternalType<oid_t>::vineyard_array_type>(
        array_builder.Seal(client));
    arrays.clear();  // release the reference

    // now use the array in vineyard memory
    auto array = local_oid_arrays_[label].GetArray();
    vineyard::HashmapBuilder<oid_t, vid_t> builder(client);
    int64_t vnum = array->length();
    builder.reserve(static_cast<size_t>(vnum));
    for (int64_t i = 0; i < vnum; ++i) {
      builder.emplace(array->GetView(i), i);
    }
    o2i_[fid_][label] =
        *std::dynamic_pointer_cast<vineyard::Hashmap<oid_t, vid_t>>(
            builder.Seal(client));

    vertices_num_[label].resize(fnum_);
    vertices_num_[label][fid_] = vnum;
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
    grape::sync_comm::AllGather(vertices_num_[label], comm_spec.comm());
  }
  return vineyard::Status::OK();
}

template <typename OID_T, typename VID_T>
vineyard::Status ArrowLocalVertexMapBuilder<OID_T, VID_T>::GetIndexOfOids(
    const std::vector<std::vector<oid_t>>& oids,
    std::vector<std::vector<vid_t>>& index_list) {
  int thread_num = std::min(
      static_cast<int>(std::thread::hardware_concurrency()), label_num_);
  index_list.resize(label_num_);
  std::atomic<int> task_id(0);
  std::vector<std::thread> threads(thread_num);
  for (int i = 0; i < thread_num; ++i) {
    threads[i] = std::thread([&]() {
      while (true) {
        int got_label_id = task_id.fetch_add(1);
        if (got_label_id >= label_num_) {
          break;
        }
        auto& o2i_map = o2i_[fid_][got_label_id];
        index_list[got_label_id].reserve(oids[got_label_id].size());
        for (const auto& oid : oids[got_label_id]) {
          auto iter = o2i_map.find(oid);
          index_list[got_label_id].push_back(iter->second);
        }
      }
    });
  }
  for (auto& thrd : threads) {
    thrd.join();
  }

  return vineyard::Status::OK();
}

template <typename OID_T, typename VID_T>
vineyard::Status
ArrowLocalVertexMapBuilder<OID_T, VID_T>::AddOuterVerticesMapping(
    std::vector<std::vector<std::vector<oid_t>>>& oids,
    std::vector<std::vector<std::vector<vid_t>>>& index_list) {
  for (fid_t fid = 0; fid < fnum_; ++fid) {
    if (fid != fid_) {
      o2i_[fid].resize(label_num_);
      i2o_[fid].resize(label_num_);
    }
  }

  auto fn = [&](fid_t cur_fid, label_id_t cur_label) -> Status {
    vineyard::HashmapBuilder<oid_t, vid_t> o2i_builder(client);
    vineyard::HashmapBuilder<vid_t, oid_t> i2o_builder(client);
    o2i_builder.reserve(static_cast<size_t>(oids[cur_fid][cur_label].size()));
    i2o_builder.reserve(static_cast<size_t>(oids[cur_fid][cur_label].size()));
    for (size_t i = 0; i < oids[cur_fid][cur_label].size(); i++) {
      auto& oid = oids[cur_fid][cur_label][i];
      auto& index = index_list[cur_fid][cur_label][i];
      o2i_builder.emplace(oid, index);
      i2o_builder.emplace(index, oid);
    }
    o2i_[cur_fid][cur_label] =
        *std::dynamic_pointer_cast<vineyard::Hashmap<oid_t, vid_t>>(
            o2i_builder.Seal(client));
    i2o_[cur_fid][cur_label] =
        *std::dynamic_pointer_cast<vineyard::Hashmap<vid_t, oid_t>>(
            i2o_builder.Seal(client));
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

template <typename VID_T>
ArrowLocalVertexMapBuilder<arrow_string_view, VID_T>::
    ArrowLocalVertexMapBuilder(vineyard::Client& client, fid_t fnum, fid_t fid,
                               label_id_t label_num)
    : client(client), fnum_(fnum), fid_(fid), label_num_(label_num) {
  oid_arrays_.resize(fnum);
  index_arrays_.resize(fnum);
  o2i_.resize(label_num);
  id_parser_.Init(fnum_, label_num_);
}

template <typename VID_T>
vineyard::Status ArrowLocalVertexMapBuilder<arrow_string_view, VID_T>::Build(
    vineyard::Client& client) {
  LOG(WARNING) << "Empty 'Build' method.";
  return vineyard::Status::OK();
}

template <typename VID_T>
std::shared_ptr<vineyard::Object>
ArrowLocalVertexMapBuilder<arrow_string_view, VID_T>::_Seal(
    vineyard::Client& client) {
  // ensure the builder hasn't been sealed yet.
  ENSURE_NOT_SEALED(this);

  auto vertex_map =
      std::make_shared<ArrowLocalVertexMap<arrow_string_view, vid_t>>();
  vertex_map->fnum_ = fnum_;
  vertex_map->fid_ = fid_;
  vertex_map->label_num_ = label_num_;
  vertex_map->id_parser_.Init(fnum_, label_num_);

  vertex_map->oid_arrays_.resize(fnum_);
  vertex_map->index_arrays_.resize(fnum_);
  for (fid_t i = 0; i < fnum_; ++i) {
    auto& array = vertex_map->oid_arrays_[i];
    auto& index = vertex_map->index_arrays_[i];
    array.resize(label_num_);
    if (i != fid_) {
      index.resize(label_num_);
    }
    for (label_id_t j = 0; j < label_num_; ++j) {
      array[j] = oid_arrays_[i][j].GetArray();
      if (i != fid_) {
        index[j] = index_arrays_[i][j].GetArray();
      }
    }
  }

  vertex_map->meta_.SetTypeName(type_name<ArrowLocalVertexMap<oid_t, vid_t>>());

  vertex_map->meta_.AddKeyValue("fnum", fnum_);
  vertex_map->meta_.AddKeyValue("fid", fid_);
  vertex_map->meta_.AddKeyValue("label_num", label_num_);

  size_t nbytes = 0;
  for (fid_t i = 0; i < fnum_; ++i) {
    for (label_id_t j = 0; j < label_num_; ++j) {
      vertex_map->meta_.AddMember(
          "oid_arrays_" + std::to_string(i) + "_" + std::to_string(j),
          oid_arrays_[i][j].meta());
      nbytes += oid_arrays_[i][j].nbytes();
      if (i != fid_) {
        vertex_map->meta_.AddMember(
            "index_arrays_" + std::to_string(i) + "_" + std::to_string(j),
            index_arrays_[i][j].meta());
        nbytes += index_arrays_[i][j].nbytes();
      }
      vertex_map->meta_.AddKeyValue(
          "vertices_num_" + std::to_string(i) + "_" + std::to_string(j),
          vertices_num_[j][i]);
    }
  }

  vertex_map->meta_.SetNBytes(nbytes);
  VINEYARD_CHECK_OK(client.CreateMetaData(vertex_map->meta_, vertex_map->id_));
  // mark the builder as sealed
  this->set_sealed(true);

  return std::static_pointer_cast<vineyard::Object>(vertex_map);
}

template <typename VID_T>
vineyard::Status
ArrowLocalVertexMapBuilder<arrow_string_view, VID_T>::AddLocalVertices(
    grape::CommSpec& comm_spec,
    std::vector<std::shared_ptr<oid_array_t>> oid_arrays) {
  std::vector<std::vector<std::shared_ptr<oid_array_t>>> arrays(
      oid_arrays.size());
  for (size_t i = 0; i < oid_arrays.size(); ++i) {
    arrays[i] = {oid_arrays[i]};
  }
  return addLocalVertices(comm_spec, std::move(arrays));
}

template <typename VID_T>
vineyard::Status
ArrowLocalVertexMapBuilder<arrow_string_view, VID_T>::AddLocalVertices(
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

template <typename VID_T>
vineyard::Status
ArrowLocalVertexMapBuilder<arrow_string_view, VID_T>::addLocalVertices(
    grape::CommSpec& comm_spec,
    std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_arrays) {
  vertices_num_.resize(label_num_);
  oid_arrays_[fid_].resize(label_num_);

  auto fn = [&](label_id_t label) -> Status {
    auto& arrays = oid_arrays[label];
    typename InternalType<oid_t>::vineyard_builder_type array_builder(client,
                                                                      arrays);
    oid_arrays_[fid_][label] = *std::dynamic_pointer_cast<
        typename InternalType<oid_t>::vineyard_array_type>(
        array_builder.Seal(client));
    arrays.clear();  // release the reference

    // now use the array in vineyard memory
    auto array = oid_arrays_[fid_][label].GetArray();
    int64_t vnum = array->length();
    o2i_[label].reserve(static_cast<size_t>(vnum));
    for (int64_t i = 0; i < vnum; ++i) {
      o2i_[label].emplace(array->GetView(i), i);
    }
    vertices_num_[label].resize(fnum_);
    vertices_num_[label][fid_] = vnum;
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
    grape::sync_comm::AllGather(vertices_num_[label], comm_spec.comm());
  }

  return vineyard::Status::OK();
}

template <typename VID_T>
vineyard::Status
ArrowLocalVertexMapBuilder<arrow_string_view, VID_T>::GetIndexOfOids(
    const std::vector<std::vector<std::string>>& oids,
    std::vector<std::vector<vid_t>>& index_list) {
  index_list.resize(label_num_);

  auto fn = [&](label_id_t label) -> Status {
    auto& o2i_map = o2i_[label];
    index_list[label].reserve(oids[label].size());
    for (const auto& oid : oids[label]) {
      index_list[label].push_back(o2i_map[oid]);
    }
    return Status::OK();
  };

  ThreadGroup tg;
  for (label_id_t label = 0; label < label_num_; ++label) {
    tg.AddTask(fn, label);
  }
  Status status;
  for (auto const& s : tg.TakeResults()) {
    status += s;
  }
  return status;
}

template <typename VID_T>
vineyard::Status
ArrowLocalVertexMapBuilder<arrow_string_view, VID_T>::AddOuterVerticesMapping(
    std::vector<std::vector<std::vector<std::string>>>& oids,
    std::vector<std::vector<std::vector<vid_t>>>& index_list) {
  for (fid_t fid = 0; fid < fnum_; ++fid) {
    if (fid != fid_) {
      oid_arrays_[fid].resize(label_num_);
      index_arrays_[fid].resize(label_num_);
    }
  }
  auto fn = [&](fid_t cur_fid, label_id_t cur_label) -> Status {
    std::shared_ptr<ArrowArrayType<std::string>> oid_array;
    std::shared_ptr<ArrowArrayType<vid_t>> index_array;
    ArrowBuilderType<std::string> array_builder;
    ArrowBuilderType<vid_t> index_builder;
    RETURN_ON_ARROW_ERROR(array_builder.AppendValues(oids[cur_fid][cur_label]));
    RETURN_ON_ARROW_ERROR(
        index_builder.AppendValues(index_list[cur_fid][cur_label]));
    RETURN_ON_ARROW_ERROR(array_builder.Finish(&oid_array));
    RETURN_ON_ARROW_ERROR(index_builder.Finish(&index_array));
    typename InternalType<oid_t>::vineyard_builder_type outer_oid_builder(
        client, oid_array);
    typename InternalType<vid_t>::vineyard_builder_type outer_index_builder(
        client, index_array);
    oid_arrays_[cur_fid][cur_label] = *std::dynamic_pointer_cast<
        typename InternalType<oid_t>::vineyard_array_type>(
        outer_oid_builder.Seal(client));
    index_arrays_[cur_fid][cur_label] = *std::dynamic_pointer_cast<
        typename InternalType<vid_t>::vineyard_array_type>(
        outer_index_builder.Seal(client));
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
