/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#ifndef MODULES_GRAPH_VERTEX_MAP_ARROW_LOCAL_VERTEX_MAP_H_
#define MODULES_GRAPH_VERTEX_MAP_ARROW_LOCAL_VERTEX_MAP_H_

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
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

namespace vineyard {

template <typename OID_T, typename VID_T>
class ArrowLocalVertexMapBuilder;

template <typename OID_T, typename VID_T>
class ArrowLocalVertexMap
    : public vineyard::Registered<ArrowLocalVertexMap<OID_T, VID_T>> {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using oid_array_t = typename vineyard::ConvertToArrowType<oid_t>::ArrayType;

 public:
  ArrowLocalVertexMap() {}
  ~ArrowLocalVertexMap() {}

  static std::unique_ptr<vineyard::Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<vineyard::Object>(
        std::unique_ptr<ArrowLocalVertexMap<OID_T, VID_T>>{
            new ArrowLocalVertexMap<OID_T, VID_T>()});
  }

  void Construct(const vineyard::ObjectMeta& meta) {
    this->meta_ = meta;
    this->id_ = meta.GetId();

    this->fnum_ = meta.GetKeyValue<fid_t>("fnum");
    this->fid_ = meta.GetKeyValue<fid_t>("fid");
    this->label_num_ = meta.GetKeyValue<label_id_t>("label_num");

    id_parser_.Init(fnum_, label_num_);

    double nbytes = 0, local_oid_total = 0, o2i_total = 0, i2o_total = 0;
    double o2i_size = 0, o2i_bucket_count = 0;
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
        o2i_[i][j].Construct(meta.GetMemberMeta("o2i_" + std::to_string(i) +
                                                "_" + std::to_string(j)));
        if (i != fid_) {
          i2o_[i][j].Construct(meta.GetMemberMeta("i2o_" + std::to_string(i) +
                                                  "_" + std::to_string(j)));
          i2o_total += i2o_[i][j].nbytes();
        }
        vertices_num_[i][j] = meta.GetKeyValue<vid_t>(
            "vertices_num_" + std::to_string(i) + "_" + std::to_string(j));

        o2i_total += o2i_[i][j].nbytes();
        o2i_size += o2i_[i][j].size();
        o2i_bucket_count += o2i_[i][j].bucket_count();
      }
    }
    nbytes = local_oid_total + o2i_total + i2o_total;
    double load_factor =
        o2i_bucket_count == 0 ? 0 : o2i_size / o2i_bucket_count;
    LOG(INFO) << "ArrowLocalVertexMap<int64_t, int64_t>\n"
              << "\tsize: " << nbytes / 1000000 << " MB\n"
              << "\to2i load factor: " << load_factor;
  }

  bool GetOid(vid_t gid, oid_t& oid) const {
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

  bool GetGid(fid_t fid, label_id_t label_id, oid_t oid, vid_t& gid) const {
    auto iter = o2i_[fid][label_id].find(oid);
    if (iter != o2i_[fid][label_id].end()) {
      gid = id_parser_.GenerateId(fid, label_id, iter->second);
      return true;
    }
    return false;
  }

  bool GetGid(label_id_t label_id, oid_t oid, vid_t& gid) const {
    for (fid_t i = 0; i < fnum_; ++i) {
      if (GetGid(i, label_id, oid, gid)) {
        return true;
      }
    }
    return false;
  }

  std::vector<oid_t> GetOids(fid_t fid, label_id_t label_id) {
    CHECK(fid == fid_);
    auto array = local_oid_arrays_[label_id];
    std::vector<oid_t> oids;

    oids.resize(array->length());
    for (auto i = 0; i < array->length(); i++) {
      oids[i] = array->GetView(i);
    }

    return oids;
  }

  std::shared_ptr<oid_array_t> GetOidArray(fid_t fid, label_id_t label_id) {
    CHECK(fid == fid_);
    return local_oid_arrays_[label_id];
  }

  fid_t fnum() { return fnum_; }

  size_t GetTotalNodesNum() const {
    size_t num = 0;
    for (auto& vec : vertices_num_) {
      for (auto& v : vec) {
        num += v;
      }
    }
    return num;
  }

  size_t GetTotalNodesNum(label_id_t label) const {
    size_t num = 0;
    for (auto& vec : vertices_num_) {
      num += vec[label];
    }
    return num;
  }

  label_id_t label_num() const { return label_num_; }

  vid_t GetInnerVertexSize(fid_t fid) const {
    size_t num = 0;
    for (auto& v : vertices_num_[fid]) {
      num += v;
    }
    return static_cast<vid_t>(num);
  }

  vid_t GetInnerVertexSize(fid_t fid, label_id_t label_id) const {
    return static_cast<vid_t>(vertices_num_[fid][label_id]);
  }

  ObjectID AddVertices(
      Client& client,
      const std::map<label_id_t, std::vector<std::shared_ptr<oid_array_t>>>&
          oid_arrays_map) {
    LOG(ERROR) << "ArrowLocalVertexMap not support AddVertices operation yet";
    return InvalidObjectID();
  }

  ObjectID AddNewVertexLabels(
      Client& client,
      const std::vector<std::vector<std::shared_ptr<oid_array_t>>>&
          oid_arrays) {
    LOG(ERROR)
        << "ArrowLocalVertexMap not support AddNewVertexLabels operation yet";
    return InvalidObjectID();
  }

 private:
  fid_t fnum_, fid_;
  label_id_t label_num_;

  vineyard::IdParser<vid_t> id_parser_;

  std::vector<std::shared_ptr<oid_array_t>> local_oid_arrays_;
  // frag->label->oid
  std::vector<std::vector<vineyard::Hashmap<oid_t, vid_t>>> o2i_;
  std::vector<std::vector<vineyard::Hashmap<vid_t, oid_t>>> i2o_;

  std::vector<std::vector<vid_t>> vertices_num_;

  friend class ArrowLocalVertexMapBuilder<OID_T, VID_T>;
};

template <typename VID_T>
class ArrowLocalVertexMap<arrow_string_view, VID_T>
    : public vineyard::Registered<
          ArrowLocalVertexMap<arrow_string_view, VID_T>> {
  using oid_t = arrow_string_view;
  using vid_t = VID_T;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using oid_array_t = arrow::LargeStringArray;
  using vid_array_t = typename vineyard::ConvertToArrowType<vid_t>::ArrayType;

 public:
  ArrowLocalVertexMap() {}
  ~ArrowLocalVertexMap() {}

  static std::unique_ptr<vineyard::Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<vineyard::Object>(
        std::unique_ptr<ArrowLocalVertexMap<oid_t, vid_t>>{
            new ArrowLocalVertexMap<oid_t, vid_t>()});
  }

  void Construct(const vineyard::ObjectMeta& meta) {
    this->meta_ = meta;
    this->id_ = meta.GetId();

    this->fnum_ = meta.GetKeyValue<fid_t>("fnum");
    this->fid_ = meta.GetKeyValue<fid_t>("fid");
    this->label_num_ = meta.GetKeyValue<label_id_t>("label_num");

    id_parser_.Init(fnum_, label_num_);

    oid_arrays_.resize(fnum_);
    index_arrays_.resize(fnum_);
    vertices_num_.resize(fnum_);
    double nbytes = 0, local_oid_total = 0, index_total = 0;
    for (fid_t i = 0; i < fnum_; ++i) {
      oid_arrays_[i].resize(label_num_);
      if (i != fid_) {
        index_arrays_[i].resize(label_num_);
      }
      vertices_num_[i].resize(label_num_);

      for (label_id_t j = 0; j < label_num_; ++j) {
        typename InternalType<oid_t>::vineyard_array_type array;
        array.Construct(meta.GetMemberMeta("oid_arrays_" + std::to_string(i) +
                                           "_" + std::to_string(j)));
        oid_arrays_[i][j] = array.GetArray();
        local_oid_total += array.nbytes();
        if (i != fid_) {
          typename InternalType<vid_t>::vineyard_array_type index_array;
          index_array.Construct(meta.GetMemberMeta(
              "index_arrays_" + std::to_string(i) + "_" + std::to_string(j)));
          index_arrays_[i][j] = index_array.GetArray();
          index_total += index_array.nbytes();
        }
        vertices_num_[i][j] = meta.GetKeyValue<vid_t>(
            "vertices_num_" + std::to_string(i) + "_" + std::to_string(j));
      }
    }

    initHashmaps();

    double o2i_size = 0, o2i_bucket_count = 0, i2o_bucket_count = 0;
    for (fid_t i = 0; i < fnum_; ++i) {
      for (label_id_t j = 0; j < label_num_; ++j) {
        if (i != fid_) {
          i2o_bucket_count += i2o_[i][j].bucket_count();
        }
        o2i_size += o2i_[i][j].size();
        o2i_bucket_count += o2i_[i][j].bucket_count();
      }
    }

    // 24 bytes = key(8) + value (8 (pointer) + 8 (length))
    nbytes = local_oid_total + index_total + o2i_bucket_count * 24 +
             i2o_bucket_count * 24;
    double load_factor =
        o2i_bucket_count == 0 ? 0 : o2i_size / o2i_bucket_count;

    LOG(INFO) << "ArrowLocalVertexMap<string, int64_t>\n"
              << "\tsize: " << nbytes / 1000000 << " MB\n"
              << "\to2i load factor: " << load_factor;
  }

  bool GetOid(vid_t gid, oid_t& oid) const {
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

  bool GetGid(fid_t fid, label_id_t label_id, oid_t oid, vid_t& gid) const {
    auto iter = o2i_[fid][label_id].find(oid);
    if (iter != o2i_[fid][label_id].end()) {
      gid = id_parser_.GenerateId(fid, label_id, iter->second);
      return true;
    }
    return false;
  }

  bool GetGid(label_id_t label_id, oid_t oid, vid_t& gid) const {
    for (fid_t i = 0; i < fnum_; ++i) {
      if (GetGid(i, label_id, oid, gid)) {
        return true;
      }
    }
    return false;
  }

  std::vector<oid_t> GetOids(fid_t fid, label_id_t label_id) {
    CHECK(fid == fid_);
    auto& array = oid_arrays_[fid][label_id];
    std::vector<oid_t> oids;

    oids.resize(array->length());
    for (auto i = 0; i < array->length(); i++) {
      oids[i] = array->GetView(i);
    }

    return oids;
  }

  fid_t fnum() { return fnum_; }

  size_t GetTotalNodesNum() const {
    size_t num = 0;
    for (auto& vec : vertices_num_) {
      for (auto& v : vec) {
        num += v;
      }
    }
    return num;
  }

  size_t GetTotalNodesNum(label_id_t label) const {
    size_t num = 0;
    for (auto& vec : vertices_num_) {
      num += vec[label];
    }
    return num;
  }

  label_id_t label_num() const { return label_num_; }

  vid_t GetInnerVertexSize(fid_t fid) const {
    size_t num = 0;
    for (auto& v : vertices_num_[fid]) {
      num += v;
    }
    return static_cast<vid_t>(num);
  }

  vid_t GetInnerVertexSize(fid_t fid, label_id_t label_id) const {
    return static_cast<vid_t>(vertices_num_[fid][label_id]);
  }

  ObjectID AddVertices(
      Client& client,
      const std::map<label_id_t, std::vector<std::shared_ptr<oid_array_t>>>&
          oid_arrays_map) {
    LOG(ERROR) << "ArrowLocalVertexMap not support AddVertices operation yet";
    return InvalidObjectID();
  }

  ObjectID AddNewVertexLabels(
      Client& client,
      const std::vector<std::vector<std::shared_ptr<oid_array_t>>>&
          oid_arrays) {
    LOG(ERROR)
        << "ArrowLocalVertexMap not support AddNewVertexLabels operation yet";
    return InvalidObjectID();
  }

 private:
  void initHashmaps() {
    o2i_.resize(fnum_);
    i2o_.resize(fnum_);
    for (fid_t i = 0; i < fnum_; ++i) {
      o2i_[i].resize(label_num_);
      if (i != fid_) {
        i2o_[i].resize(label_num_);
      }
    }

    int task_num = static_cast<int>(fnum_) * static_cast<int>(label_num_);
    int thread_num = std::min(
        static_cast<int>(std::thread::hardware_concurrency()), task_num);
    std::atomic<int> task_id(0);
    std::vector<std::thread> threads(thread_num);
    for (int i = 0; i < thread_num; ++i) {
      threads[i] = std::thread([&]() {
        while (true) {
          int got_task_id = task_id.fetch_add(1);
          if (got_task_id >= task_num) {
            break;
          }
          fid_t cur_fid = static_cast<fid_t>(got_task_id) % fnum_;
          label_id_t cur_label =
              static_cast<label_id_t>(static_cast<fid_t>(got_task_id) / fnum_);
          int64_t vnum = oid_arrays_[cur_fid][cur_label]->length();
          if (cur_fid == fid_) {
            for (int64_t i = 0; i < vnum; ++i) {
              auto oid = oid_arrays_[cur_fid][cur_label]->GetView(i);
              o2i_[cur_fid][cur_label].emplace(oid, i);
            }
          } else {
            for (int64_t i = 0; i < vnum; ++i) {
              auto oid = oid_arrays_[cur_fid][cur_label]->GetView(i);
              auto index = index_arrays_[cur_fid][cur_label]->GetView(i);
              o2i_[cur_fid][cur_label].emplace(oid, index);
              i2o_[cur_fid][cur_label].emplace(index, oid);
            }
          }
        }
      });
    }
    for (auto& thrd : threads) {
      thrd.join();
    }
  }

  fid_t fnum_, fid_;
  label_id_t label_num_;

  vineyard::IdParser<vid_t> id_parser_;

  // frag->label->oid
  std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_arrays_;
  std::vector<std::vector<std::shared_ptr<vid_array_t>>> index_arrays_;
  std::vector<std::vector<ska::flat_hash_map<oid_t, vid_t>>> o2i_;
  std::vector<std::vector<ska::flat_hash_map<vid_t, oid_t>>> i2o_;

  std::vector<std::vector<vid_t>> vertices_num_;

  friend class ArrowLocalVertexMapBuilder<arrow_string_view, VID_T>;
};

template <typename OID_T, typename VID_T>
class ArrowLocalVertexMapBuilder : public vineyard::ObjectBuilder {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using oid_array_t = typename vineyard::ConvertToArrowType<oid_t>::ArrayType;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;

 public:
  explicit ArrowLocalVertexMapBuilder(vineyard::Client& client)
      : client(client) {}

  explicit ArrowLocalVertexMapBuilder(vineyard::Client& client, fid_t fnum,
                                      fid_t fid, label_id_t label_num)
      : client(client), fnum_(fnum), fid_(fid), label_num_(label_num) {
    local_oid_arrays_.resize(label_num);
    o2i_.resize(fnum);
    o2i_[fid].resize(label_num);
    i2o_.resize(fnum);
    id_parser_.Init(fnum_, label_num_);
  }

  vineyard::Status Build(vineyard::Client& client) {
    LOG(ERROR) << "Not implement Build method.";
    return vineyard::Status::OK();
  }

  std::shared_ptr<vineyard::Object> _Seal(vineyard::Client& client) {
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

    vertex_map->meta_.SetTypeName(
        type_name<ArrowLocalVertexMap<oid_t, vid_t>>());

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

    VINEYARD_CHECK_OK(
        client.CreateMetaData(vertex_map->meta_, vertex_map->id_));
    // mark the builder as sealed
    this->set_sealed(true);

    return std::static_pointer_cast<vineyard::Object>(vertex_map);
  }

  vineyard::Status AddLocalVertices(
      grape::CommSpec& comm_spec,
      std::vector<std::shared_ptr<oid_array_t>> oid_arrays) {
    int thread_num = std::min(
        static_cast<int>(std::thread::hardware_concurrency()), label_num_);
    std::atomic<int> task_id(0);
    std::vector<std::thread> threads(thread_num);
    vertices_num_.resize(label_num_);
    for (int i = 0; i < thread_num; ++i) {
      threads[i] = std::thread([&]() {
        while (true) {
          int got_label_id = task_id.fetch_add(1);
          if (got_label_id >= label_num_) {
            break;
          }
          auto& array = oid_arrays[got_label_id];
          typename InternalType<oid_t>::vineyard_builder_type array_builder(
              client, array);
          local_oid_arrays_[got_label_id] = *std::dynamic_pointer_cast<
              typename InternalType<oid_t>::vineyard_array_type>(
              array_builder.Seal(client));
          vineyard::HashmapBuilder<oid_t, vid_t> builder(client);
          int64_t vnum = oid_arrays[got_label_id]->length();
          for (int64_t i = 0; i < vnum; ++i) {
            builder.emplace(oid_arrays[got_label_id]->GetView(i), i);
          }
          o2i_[fid_][got_label_id] =
              *std::dynamic_pointer_cast<vineyard::Hashmap<oid_t, vid_t>>(
                  builder.Seal(client));

          vertices_num_[got_label_id].resize(fnum_);
          vertices_num_[got_label_id][fid_] = vnum;
        }
      });
    }
    for (auto& thrd : threads) {
      thrd.join();
    }

    // sync the vertices_num
    for (label_id_t label = 0; label < label_num_; ++label) {
      grape::sync_comm::AllGather(vertices_num_[label], comm_spec.comm());
    }
    return vineyard::Status::OK();
  }

  vineyard::Status GetIndexOfOids(const std::vector<std::vector<oid_t>>& oids,
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

  vineyard::Status AddOuterVerticesMapping(
      std::vector<std::vector<std::vector<oid_t>>>& oids,
      std::vector<std::vector<std::vector<vid_t>>>& index_list) {
    int task_num = static_cast<int>(fnum_) * static_cast<int>(label_num_);
    int thread_num = std::min(
        static_cast<int>(std::thread::hardware_concurrency()), task_num);
    std::atomic<int> task_id(0);
    std::vector<std::thread> threads(thread_num);
    for (fid_t i = 0; i < fnum_; ++i) {
      if (i != fid_) {
        o2i_[i].resize(label_num_);
        i2o_[i].resize(label_num_);
      }
    }
    for (int i = 0; i < thread_num; ++i) {
      threads[i] = std::thread([&]() {
        while (true) {
          int got_task_id = task_id.fetch_add(1);
          if (got_task_id >= task_num) {
            break;
          }
          fid_t cur_fid = static_cast<fid_t>(got_task_id) % fnum_;
          if (cur_fid == fid_) {
            continue;
          }
          label_id_t cur_label =
              static_cast<label_id_t>(static_cast<fid_t>(got_task_id) / fnum_);
          vineyard::HashmapBuilder<oid_t, vid_t> o2i_builder(client);
          vineyard::HashmapBuilder<vid_t, oid_t> i2o_builder(client);
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
        }
      });
    }
    for (auto& thrd : threads) {
      thrd.join();
    }

    return vineyard::Status::OK();
  }

 private:
  vineyard::Client& client;
  fid_t fnum_, fid_;
  label_id_t label_num_;

  vineyard::IdParser<vid_t> id_parser_;

  std::vector<typename InternalType<oid_t>::vineyard_array_type>
      local_oid_arrays_;
  std::vector<std::vector<vineyard::Hashmap<oid_t, vid_t>>> o2i_;
  std::vector<std::vector<vineyard::Hashmap<vid_t, oid_t>>> i2o_;
  std::vector<std::vector<vid_t>> vertices_num_;
};

template <typename VID_T>
class ArrowLocalVertexMapBuilder<arrow_string_view, VID_T>
    : public vineyard::ObjectBuilder {
  using oid_t = arrow_string_view;
  using vid_t = VID_T;
  using oid_array_t = arrow::LargeStringArray;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;

 public:
  explicit ArrowLocalVertexMapBuilder(vineyard::Client& client)
      : client(client) {}

  explicit ArrowLocalVertexMapBuilder(vineyard::Client& client, fid_t fnum,
                                      fid_t fid, label_id_t label_num)
      : client(client), fnum_(fnum), fid_(fid), label_num_(label_num) {
    oid_arrays_.resize(fnum);
    index_arrays_.resize(fnum);
    o2i_.resize(label_num);
    id_parser_.Init(fnum_, label_num_);
  }

  vineyard::Status Build(vineyard::Client& client) {
    LOG(ERROR) << "Not implement Build method.";
    return vineyard::Status::OK();
  }

  std::shared_ptr<vineyard::Object> _Seal(vineyard::Client& client) {
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

    vertex_map->meta_.SetTypeName(
        type_name<ArrowLocalVertexMap<oid_t, vid_t>>());

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
    VINEYARD_CHECK_OK(
        client.CreateMetaData(vertex_map->meta_, vertex_map->id_));
    // mark the builder as sealed
    this->set_sealed(true);

    return std::static_pointer_cast<vineyard::Object>(vertex_map);
  }

  vineyard::Status AddLocalVertices(
      grape::CommSpec& comm_spec,
      std::vector<std::shared_ptr<oid_array_t>> oid_arrays) {
    int thread_num = std::min(
        static_cast<int>(std::thread::hardware_concurrency()), label_num_);
    std::atomic<int> task_id(0);
    vertices_num_.resize(label_num_);
    std::vector<std::thread> threads(thread_num);
    oid_arrays_[fid_].resize(label_num_);
    for (int i = 0; i < thread_num; ++i) {
      threads[i] = std::thread([&]() {
        while (true) {
          int got_label_id = task_id.fetch_add(1);
          if (got_label_id >= label_num_) {
            break;
          }
          auto& array = oid_arrays[got_label_id];
          typename InternalType<oid_t>::vineyard_builder_type array_builder(
              client, array);
          oid_arrays_[fid_][got_label_id] = *std::dynamic_pointer_cast<
              typename InternalType<oid_t>::vineyard_array_type>(
              array_builder.Seal(client));
          int64_t vnum = array->length();
          for (int64_t i = 0; i < vnum; ++i) {
            o2i_[got_label_id].emplace(array->GetView(i), i);
          }
          vertices_num_[got_label_id].resize(fnum_);
          vertices_num_[got_label_id][fid_] = vnum;
        }
      });
    }
    for (auto& thrd : threads) {
      thrd.join();
    }

    // sync the vertices_num
    for (label_id_t label = 0; label < label_num_; ++label) {
      grape::sync_comm::AllGather(vertices_num_[label], comm_spec.comm());
    }

    return vineyard::Status::OK();
  }

  vineyard::Status GetIndexOfOids(
      const std::vector<std::vector<std::string>>& oids,
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
          auto& o2i_map = o2i_[got_label_id];
          index_list[got_label_id].reserve(oids[got_label_id].size());
          for (const auto& oid : oids[got_label_id]) {
            index_list[got_label_id].push_back(o2i_map[oid]);
          }
        }
      });
    }
    for (auto& thrd : threads) {
      thrd.join();
    }

    return vineyard::Status::OK();
  }

  vineyard::Status AddOuterVerticesMapping(
      std::vector<std::vector<std::vector<std::string>>>& oids,
      std::vector<std::vector<std::vector<vid_t>>>& index_list) {
    int task_num = static_cast<int>(fnum_) * static_cast<int>(label_num_);
    int thread_num = std::min(
        static_cast<int>(std::thread::hardware_concurrency()), task_num);
    std::atomic<int> task_id(0);
    std::vector<std::thread> threads(thread_num);
    for (fid_t i = 0; i < fnum_; ++i) {
      if (i != fid_) {
        oid_arrays_[i].resize(label_num_);
        index_arrays_[i].resize(label_num_);
      }
    }
    std::vector<vineyard::Status> status(thread_num);
    for (int i = 0; i < thread_num; ++i) {
      threads[i] = std::thread(
          [&](Status& status) -> void {
            while (true) {
              int got_task_id = task_id.fetch_add(1);
              if (got_task_id >= task_num) {
                break;
              }
              fid_t cur_fid = static_cast<fid_t>(got_task_id) % fnum_;
              if (cur_fid == fid_) {
                continue;
              }
              label_id_t cur_label = static_cast<label_id_t>(
                  static_cast<fid_t>(got_task_id) / fnum_);
              std::shared_ptr<
                  typename ConvertToArrowType<std::string>::ArrayType>
                  oid_array;
              std::shared_ptr<typename ConvertToArrowType<vid_t>::ArrayType>
                  index_array;
              typename ConvertToArrowType<std::string>::BuilderType
                  array_builder;
              typename ConvertToArrowType<vid_t>::BuilderType index_builder;
              status += vineyard::Status::ArrowError(
                  array_builder.AppendValues(oids[cur_fid][cur_label]));
              if (!status.ok()) {
                return;
              }
              status += vineyard::Status::ArrowError(
                  index_builder.AppendValues(index_list[cur_fid][cur_label]));
              if (!status.ok()) {
                return;
              }
              status += vineyard::Status::ArrowError(
                  array_builder.Finish(&oid_array));
              if (!status.ok()) {
                return;
              }
              status += vineyard::Status::ArrowError(
                  index_builder.Finish(&index_array));
              if (!status.ok()) {
                return;
              }
              typename InternalType<oid_t>::vineyard_builder_type
                  outer_oid_builder(client, oid_array);
              typename InternalType<vid_t>::vineyard_builder_type
                  outer_index_builder(client, index_array);
              oid_arrays_[cur_fid][cur_label] = *std::dynamic_pointer_cast<
                  typename InternalType<oid_t>::vineyard_array_type>(
                  outer_oid_builder.Seal(client));
              index_arrays_[cur_fid][cur_label] = *std::dynamic_pointer_cast<
                  typename InternalType<vid_t>::vineyard_array_type>(
                  outer_index_builder.Seal(client));
            }
          },
          std::ref(status[i]));
    }
    for (auto& thrd : threads) {
      thrd.join();
    }
    auto ret = vineyard::Status::OK();
    for (auto& st : status) {
      ret += st;
    }
    return ret;
  }

 private:
  vineyard::Client& client;
  fid_t fnum_, fid_;
  label_id_t label_num_;

  vineyard::IdParser<vid_t> id_parser_;

  std::vector<std::vector<typename InternalType<oid_t>::vineyard_array_type>>
      oid_arrays_;
  std::vector<std::vector<typename InternalType<vid_t>::vineyard_array_type>>
      index_arrays_;
  std::vector<ska::flat_hash_map<oid_t, vid_t>> o2i_;
  std::vector<std::vector<vid_t>> vertices_num_;
};

template <typename T>
struct is_local_vertex_map {
  using type = std::false_type;
  static constexpr bool value = false;
};

template <typename OID_T, typename VID_T>
struct is_local_vertex_map<ArrowLocalVertexMap<OID_T, VID_T>> {
  using type = std::true_type;
  static constexpr bool value = true;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_VERTEX_MAP_ARROW_LOCAL_VERTEX_MAP_H_
