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

#ifndef MODULES_GRAPH_VERTEX_MAP_ARROW_VERTEX_MAP_BUILDER_H_
#define MODULES_GRAPH_VERTEX_MAP_ARROW_VERTEX_MAP_BUILDER_H_

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "basic/ds/arrow.h"
#include "basic/ds/hashmap.h"
#include "client/client.h"
#include "common/util/functions.h"
#include "common/util/typename.h"

#include "graph/fragment/property_graph_types.h"
#include "graph/utils/thread_group.h"
#include "graph/vertex_map/arrow_vertex_map.h"

namespace vineyard {

template <typename OID_T, typename VID_T>
ObjectID ArrowVertexMap<OID_T, VID_T>::AddVertices(
    Client& client,
    const std::map<label_id_t, std::vector<std::shared_ptr<oid_array_t>>>&
        oid_arrays_map) {
  int extra_label_num = oid_arrays_map.size();

  std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_arrays;
  oid_arrays.resize(extra_label_num);
  for (auto& pair : oid_arrays_map) {
    oid_arrays[pair.first - label_num_] = pair.second;
  }
  return AddNewVertexLabels(client, oid_arrays);
}

template <typename OID_T, typename VID_T>
ObjectID ArrowVertexMap<OID_T, VID_T>::AddNewVertexLabels(
    Client& client,
    const std::vector<std::vector<std::shared_ptr<oid_array_t>>>& oid_arrays) {
  size_t extra_label_num = oid_arrays.size();
  int task_num = static_cast<int>(fnum_) * static_cast<int>(extra_label_num);

  std::vector<std::vector<typename InternalType<oid_t>::vineyard_array_type>>
      vy_oid_arrays;
  std::vector<std::vector<vineyard::Hashmap<oid_t, vid_t>>> vy_o2g;
  int total_label_num = label_num_ + extra_label_num;
  vy_oid_arrays.resize(fnum_);
  vy_o2g.resize(fnum_);
  for (fid_t i = 0; i < fnum_; ++i) {
    vy_oid_arrays[i].resize(extra_label_num);
    vy_o2g[i].resize(extra_label_num);
  }

  int thread_num =
      std::min(static_cast<int>(std::thread::hardware_concurrency()), task_num);
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
        auto cur_label =
            static_cast<label_id_t>(static_cast<fid_t>(got_task_id) / fnum_);

        vineyard::HashmapBuilder<oid_t, vid_t> builder(client);
        auto array = oid_arrays[cur_label][cur_fid];
        {
          vid_t cur_gid =
              id_parser_.GenerateId(cur_fid, label_num_ + cur_label, 0);
          int64_t vnum = array->length();
          // builder.reserve(static_cast<size_t>(vnum));
          for (int64_t k = 0; k < vnum; ++k) {
            builder.emplace(array->GetView(k), cur_gid);
            ++cur_gid;
          }
        }

        {
          typename InternalType<oid_t>::vineyard_builder_type array_builder(
              client, array);
          vy_oid_arrays[cur_fid][cur_label] =
              *std::dynamic_pointer_cast<vineyard::NumericArray<oid_t>>(
                  array_builder.Seal(client));

          vy_o2g[cur_fid][cur_label] =
              *std::dynamic_pointer_cast<vineyard::Hashmap<oid_t, vid_t>>(
                  builder.Seal(client));
        }
      }
    });
  }
  for (auto& thrd : threads) {
    thrd.join();
  }

  vineyard::ObjectMeta old_meta, new_meta;
  VINEYARD_CHECK_OK(client.GetMetaData(this->id(), old_meta));

  new_meta.SetTypeName(type_name<ArrowVertexMap<oid_t, vid_t>>());

  new_meta.AddKeyValue("fnum", fnum_);
  new_meta.AddKeyValue("label_num", total_label_num);

  size_t nbytes = 0;
  for (fid_t i = 0; i < fnum_; ++i) {
    for (label_id_t j = 0; j < total_label_num; ++j) {
      std::string array_name =
          "oid_arrays_" + std::to_string(i) + "_" + std::to_string(j);
      std::string map_name =
          "o2g_" + std::to_string(i) + "_" + std::to_string(j);
      if (j < label_num_) {
        auto array_meta = old_meta.GetMemberMeta(array_name);
        new_meta.AddMember(array_name, array_meta);
        nbytes += array_meta.GetNBytes();

        auto map_meta = old_meta.GetMemberMeta(map_name);
        new_meta.AddMember(map_name, map_meta);
        nbytes += map_meta.GetNBytes();
      } else {
        new_meta.AddMember(array_name, vy_oid_arrays[i][j - label_num_].meta());
        nbytes += vy_oid_arrays[i][j - label_num_].nbytes();

        new_meta.AddMember(map_name, vy_o2g[i][j - label_num_].meta());
        nbytes += vy_o2g[i][j - label_num_].nbytes();
      }
    }
  }

  new_meta.SetNBytes(nbytes);
  ObjectID ret;
  VINEYARD_CHECK_OK(client.CreateMetaData(new_meta, ret));
  VLOG(2) << "vertex map memory usage: " << new_meta.MemoryUsage() << " bytes";
  return ret;
}

template <typename VID_T>
ObjectID ArrowVertexMap<arrow_string_view, VID_T>::AddVertices(
    Client& client,
    const std::map<label_id_t, std::vector<std::shared_ptr<oid_array_t>>>&
        oid_arrays_map) {
  int extra_label_num = oid_arrays_map.size();

  std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_arrays;
  oid_arrays.resize(extra_label_num);
  for (auto& pair : oid_arrays_map) {
    oid_arrays[pair.first - label_num_] = pair.second;
  }
  return AddNewVertexLabels(client, oid_arrays);
}

template <typename VID_T>
ObjectID ArrowVertexMap<arrow_string_view, VID_T>::AddNewVertexLabels(
    Client& client,
    const std::vector<std::vector<std::shared_ptr<oid_array_t>>>& oid_arrays) {
  size_t extra_label_num = oid_arrays.size();

  std::vector<std::vector<typename InternalType<oid_t>::vineyard_array_type>>
      vy_oid_arrays;
  int total_label_num = label_num_ + extra_label_num;
  vy_oid_arrays.resize(fnum_);
  for (fid_t i = 0; i < fnum_; ++i) {
    vy_oid_arrays[i].resize(extra_label_num);
  }

  ThreadGroup tg;
  auto builder_fn = [&client, &oid_arrays, &vy_oid_arrays](
                        fid_t const fid, label_id_t const vlabel_id) -> Status {
    auto& array = oid_arrays[vlabel_id][fid];
    typename InternalType<oid_t>::vineyard_builder_type array_builder(client,
                                                                      array);
    vy_oid_arrays[fid][vlabel_id] = *std::dynamic_pointer_cast<
        typename InternalType<oid_t>::vineyard_array_type>(
        array_builder.Seal(client));
    return Status::OK();
  };

  for (fid_t fid = 0; fid < fnum_; ++fid) {
    for (size_t vlabel_id = 0; vlabel_id < extra_label_num; ++vlabel_id) {
      tg.AddTask(builder_fn, fid, vlabel_id);
    }
  }
  tg.TakeResults();

  vineyard::ObjectMeta old_meta, new_meta;
  VINEYARD_CHECK_OK(client.GetMetaData(this->id(), old_meta));

  new_meta.SetTypeName(type_name<ArrowVertexMap<oid_t, vid_t>>());

  new_meta.AddKeyValue("fnum", fnum_);
  new_meta.AddKeyValue("label_num", total_label_num);

  size_t nbytes = 0;
  for (fid_t i = 0; i < fnum_; ++i) {
    for (label_id_t j = 0; j < total_label_num; ++j) {
      std::string array_name =
          "oid_arrays_" + std::to_string(i) + "_" + std::to_string(j);
      if (j < label_num_) {
        auto array_meta = old_meta.GetMemberMeta(array_name);
        new_meta.AddMember(array_name, array_meta);
        nbytes += array_meta.GetNBytes();
      } else {
        new_meta.AddMember(array_name, vy_oid_arrays[i][j - label_num_].meta());
        nbytes += vy_oid_arrays[i][j - label_num_].nbytes();
      }
    }
  }
  new_meta.SetNBytes(nbytes);
  ObjectID ret;
  VINEYARD_CHECK_OK(client.CreateMetaData(new_meta, ret));
  VLOG(2) << "vertex map memory usage: " << new_meta.MemoryUsage() << " bytes";
  return ret;
}

template <typename OID_T, typename VID_T>
std::shared_ptr<vineyard::Object> ArrowVertexMapBuilder<OID_T, VID_T>::_Seal(
    vineyard::Client& client) {
  // ensure the builder hasn't been sealed yet.
  ENSURE_NOT_SEALED(this);

  VINEYARD_CHECK_OK(this->Build(client));

  auto vertex_map = std::make_shared<ArrowVertexMap<oid_t, vid_t>>();
  vertex_map->fnum_ = fnum_;
  vertex_map->label_num_ = label_num_;
  vertex_map->id_parser_.Init(fnum_, label_num_);

  vertex_map->oid_arrays_.resize(fnum_);
  for (fid_t i = 0; i < fnum_; ++i) {
    auto& array = vertex_map->oid_arrays_[i];
    array.resize(label_num_);
    for (label_id_t j = 0; j < label_num_; ++j) {
      array[j] = oid_arrays_[i][j].GetArray();
    }
  }

  vertex_map->o2g_ = o2g_;

  vertex_map->meta_.SetTypeName(type_name<ArrowVertexMap<oid_t, vid_t>>());

  vertex_map->meta_.AddKeyValue("fnum", fnum_);
  vertex_map->meta_.AddKeyValue("label_num", label_num_);

  size_t nbytes = 0;
  for (fid_t i = 0; i < fnum_; ++i) {
    for (label_id_t j = 0; j < label_num_; ++j) {
      vertex_map->meta_.AddMember(
          "oid_arrays_" + std::to_string(i) + "_" + std::to_string(j),
          oid_arrays_[i][j].meta());
      nbytes += oid_arrays_[i][j].nbytes();

      vertex_map->meta_.AddMember(
          "o2g_" + std::to_string(i) + "_" + std::to_string(j),
          o2g_[i][j].meta());
      nbytes += o2g_[i][j].nbytes();
    }
  }

  vertex_map->meta_.SetNBytes(nbytes);

  VINEYARD_CHECK_OK(client.CreateMetaData(vertex_map->meta_, vertex_map->id_));
  VLOG(2) << "vertex map memory usage: " << vertex_map->meta_.MemoryUsage()
          << " bytes";

  // mark the builder as sealed
  this->set_sealed(true);

  return std::static_pointer_cast<vineyard::Object>(vertex_map);
}

template <typename VID_T>
std::shared_ptr<vineyard::Object>
ArrowVertexMapBuilder<arrow_string_view, VID_T>::_Seal(
    vineyard::Client& client) {
  // ensure the builder hasn't been sealed yet.
  ENSURE_NOT_SEALED(this);

  VINEYARD_CHECK_OK(this->Build(client));

  auto vertex_map =
      std::make_shared<ArrowVertexMap<arrow_string_view, vid_t>>();
  vertex_map->fnum_ = fnum_;
  vertex_map->label_num_ = label_num_;
  vertex_map->id_parser_.Init(fnum_, label_num_);

  vertex_map->oid_arrays_.resize(fnum_);
  for (fid_t i = 0; i < fnum_; ++i) {
    auto& array = vertex_map->oid_arrays_[i];
    array.resize(label_num_);
    for (label_id_t j = 0; j < label_num_; ++j) {
      array[j] = oid_arrays_[i][j].GetArray();
    }
  }

  vertex_map->meta_.SetTypeName(type_name<ArrowVertexMap<oid_t, vid_t>>());

  vertex_map->meta_.AddKeyValue("fnum", fnum_);
  vertex_map->meta_.AddKeyValue("label_num", label_num_);

  size_t nbytes = 0;
  for (fid_t i = 0; i < fnum_; ++i) {
    for (label_id_t j = 0; j < label_num_; ++j) {
      vertex_map->meta_.AddMember(
          "oid_arrays_" + std::to_string(i) + "_" + std::to_string(j),
          oid_arrays_[i][j].meta());
      nbytes += oid_arrays_[i][j].nbytes();
    }
  }

  vertex_map->meta_.SetNBytes(nbytes);
  VINEYARD_CHECK_OK(client.CreateMetaData(vertex_map->meta_, vertex_map->id_));
  VLOG(2) << "vertex map memory usage: " << vertex_map->meta_.MemoryUsage()
          << " bytes";

  // mark the builder as sealed
  this->set_sealed(true);

  return std::static_pointer_cast<vineyard::Object>(vertex_map);
}

template <typename OID_T, typename VID_T>
vineyard::Status BasicArrowVertexMapBuilder<OID_T, VID_T>::Build(
    vineyard::Client& client) {
  this->set_fnum_label_num(fnum_, label_num_);

#if 0
    for (fid_t i = 0; i < fnum_; ++i) {
      // TODO(luoxiaojian): parallel construct hashmap
      for (label_id_t j = 0; j < label_num_; ++j) {
        vineyard::HashmapBuilder<oid_t, vid_t> builder(client);

        auto array = oid_arrays_[j][i];
        {
          vid_t cur_gid = id_parser_.GenerateId(i, j, 0);
          int64_t vnum = array->length();
          for (int64_t k = 0; k < vnum; ++k) {
            builder.emplace(array->GetView(k), cur_gid);
            ++cur_gid;
          }
        }

        typename InternalType<oid_t>::vineyard_builder_type array_builder(
            client, array);
        this->set_oid_array(
            i, j,
            *std::dynamic_pointer_cast<vineyard::NumericArray<oid_t>>(
                array_builder.Seal(client)));

        this->set_o2g(
            i, j,
            *std::dynamic_pointer_cast<vineyard::Hashmap<oid_t, vid_t>>(
                builder.Seal(client)));
      }
    }
#else
  int task_num = static_cast<int>(fnum_) * static_cast<int>(label_num_);
  int thread_num =
      std::min(static_cast<int>(std::thread::hardware_concurrency()), task_num);
  std::atomic<int> task_id(0);

#if defined(WITH_PROFILING)
  auto start_ts = GetCurrentTime();
#endif

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

        vineyard::HashmapBuilder<oid_t, vid_t> builder(client);
        auto array = oid_arrays_[cur_label][cur_fid];
        {
          vid_t cur_gid = id_parser_.GenerateId(cur_fid, cur_label, 0);
          int64_t vnum = array->length();
          // builder.reserve(static_cast<size_t>(vnum));
          for (int64_t k = 0; k < vnum; ++k) {
            builder.emplace(array->GetView(k), cur_gid);
            ++cur_gid;
          }
        }

        {
          typename InternalType<oid_t>::vineyard_builder_type array_builder(
              client, array);
          this->set_oid_array(
              cur_fid, cur_label,
              *std::dynamic_pointer_cast<vineyard::NumericArray<oid_t>>(
                  array_builder.Seal(client)));

          this->set_o2g(
              cur_fid, cur_label,
              *std::dynamic_pointer_cast<vineyard::Hashmap<oid_t, vid_t>>(
                  builder.Seal(client)));
        }
      }
    });
  }
  for (auto& thrd : threads) {
    thrd.join();
  }

#if defined(WITH_PROFILING)
  auto finish_seal_ts = GetCurrentTime();
  LOG(INFO) << "Seal hashmaps uses " << (finish_seal_ts - start_ts)
            << " seconds";
#endif

#endif

  return vineyard::Status::OK();
}

template <typename VID_T>
vineyard::Status BasicArrowVertexMapBuilder<arrow_string_view, VID_T>::Build(
    vineyard::Client& client) {
  this->set_fnum_label_num(fnum_, label_num_);

  ThreadGroup tg;
  auto builder_fn = [this, &client](fid_t const fid,
                                    label_id_t const vlabel_id) -> Status {
    auto& array = oid_arrays_[vlabel_id][fid];
    typename InternalType<oid_t>::vineyard_builder_type array_builder(client,
                                                                      array);
    this->set_oid_array(fid, vlabel_id,
                        *std::dynamic_pointer_cast<
                            typename InternalType<oid_t>::vineyard_array_type>(
                            array_builder.Seal(client)));
    return Status::OK();
  };

  for (fid_t fid = 0; fid < fnum_; ++fid) {
    for (label_id_t vlabel_id = 0; vlabel_id < label_num_; ++vlabel_id) {
      tg.AddTask(builder_fn, fid, vlabel_id);
    }
  }
  tg.TakeResults();
  return vineyard::Status::OK();
}

}  // namespace vineyard

#endif  // MODULES_GRAPH_VERTEX_MAP_ARROW_VERTEX_MAP_BUILDER_H_
