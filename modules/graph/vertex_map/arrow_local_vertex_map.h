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

namespace grape {
class CommSpec;
}  // namespace grape

namespace vineyard {

template <typename OID_T, typename VID_T>
class ArrowLocalVertexMapBuilder;

template <typename OID_T, typename VID_T>
class ArrowLocalVertexMap
    : public vineyard::Registered<ArrowLocalVertexMap<OID_T, VID_T>> {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using oid_array_t = ArrowArrayType<oid_t>;

  static_assert(!std::is_same<OID_T, std::string>::value,
                "Expect arrow_string_view in local vertex map's OID_T");

 public:
  ArrowLocalVertexMap() {}
  ~ArrowLocalVertexMap() {}

  static std::unique_ptr<vineyard::Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<vineyard::Object>(
        std::unique_ptr<ArrowLocalVertexMap<OID_T, VID_T>>{
            new ArrowLocalVertexMap<OID_T, VID_T>()});
  }

  void Construct(const vineyard::ObjectMeta& meta);

  bool GetOid(vid_t gid, oid_t& oid) const;

  bool GetGid(fid_t fid, label_id_t label_id, oid_t oid, vid_t& gid) const;

  bool GetGid(label_id_t label_id, oid_t oid, vid_t& gid) const;

  std::vector<OID_T> GetOids(fid_t fid, label_id_t label_id) const;

  std::shared_ptr<oid_array_t> GetOidArray(fid_t fid,
                                           label_id_t label_id) const;

  fid_t fnum() { return fnum_; }

  bool use_perfect_hash() const { return false; }

  size_t GetTotalNodesNum() const;

  size_t GetTotalNodesNum(label_id_t label) const;

  label_id_t label_num() const { return label_num_; }

  VID_T GetInnerVertexSize(fid_t fid) const;

  VID_T GetInnerVertexSize(fid_t fid, label_id_t label_id) const;

  ObjectID AddVertices(
      Client& client,
      const std::map<label_id_t, std::vector<std::shared_ptr<oid_array_t>>>&
          oid_arrays_map);

  ObjectID AddVertices(
      Client& client,
      const std::map<label_id_t,
                     std::vector<std::shared_ptr<arrow::ChunkedArray>>>&
          oid_arrays_map);

  ObjectID AddNewVertexLabels(
      Client& client,
      const std::vector<std::vector<std::shared_ptr<oid_array_t>>>& oid_arrays);

  ObjectID AddNewVertexLabels(
      Client& client,
      const std::vector<std::vector<std::shared_ptr<arrow::ChunkedArray>>>&
          oid_arrays);

 private:
  fid_t fnum_, fid_;
  label_id_t label_num_;

  vineyard::IdParser<vid_t> id_parser_;

  // frag->label->oid
  //
  // when oid_t is not string_view, only current fragment's is non-empty
  std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_arrays_;

  // frag->label->oid
  std::vector<std::vector<vineyard::Hashmap<oid_t, vid_t>>> o2i_;

  // for non-string_view oid type
  std::vector<std::vector<vineyard::Hashmap<vid_t, oid_t>>> i2o_;
  // for string_view oid_type
  std::vector<std::vector<vineyard::Hashmap<vid_t, vid_t /* index */>>>
      i2o_index_;

  std::vector<std::vector<vid_t>> vertices_num_;

  friend class ArrowLocalVertexMapBuilder<OID_T, VID_T>;
};

template <typename OID_T, typename VID_T>
class ArrowLocalVertexMapBuilder : public vineyard::ObjectBuilder {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using oid_array_t = ArrowArrayType<oid_t>;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;

  static_assert(!std::is_same<OID_T, std::string>::value,
                "Expect arrow_string_view in local vertex map's OID_T");

 public:
  explicit ArrowLocalVertexMapBuilder(vineyard::Client& client)
      : client(client) {}

  explicit ArrowLocalVertexMapBuilder(vineyard::Client& client, fid_t fnum,
                                      fid_t fid, label_id_t label_num);

  vineyard::Status Build(vineyard::Client& client) override;

  Status _Seal(vineyard::Client& client,
               std::shared_ptr<vineyard::Object>& object) override;

  vineyard::Status AddLocalVertices(
      grape::CommSpec& comm_spec,
      std::vector<std::shared_ptr<oid_array_t>> oid_arrays);

  vineyard::Status AddLocalVertices(
      grape::CommSpec& comm_spec,
      std::vector<std::shared_ptr<arrow::ChunkedArray>> oid_arrays);

  vineyard::Status GetIndexOfOids(
      const std::vector<std::shared_ptr<oid_array_t>>& oids,
      std::vector<std::vector<vid_t>>& index_list);

  template <typename OID_TYPE,
            typename std::enable_if<!std::is_same<
                OID_TYPE, arrow_string_view>::value>::type* = nullptr>
  vineyard::Status AddOuterVerticesMapping(
      std::vector<std::vector<std::shared_ptr<ArrowArrayType<OID_TYPE>>>> oids,
      std::vector<std::vector<std::vector<vid_t>>> index_list);

  template <typename OID_TYPE,
            typename std::enable_if<std::is_same<
                OID_TYPE, arrow_string_view>::value>::type* = nullptr>
  vineyard::Status AddOuterVerticesMapping(
      std::vector<std::vector<std::shared_ptr<ArrowArrayType<OID_TYPE>>>> oids,
      std::vector<std::vector<std::vector<vid_t>>> index_list);

 private:
  vineyard::Status addLocalVertices(
      grape::CommSpec& comm_spec,
      std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_arrays);

  vineyard::Client& client;
  fid_t fnum_, fid_;
  label_id_t label_num_;

  vineyard::IdParser<vid_t> id_parser_;

  std::vector<typename InternalType<oid_t>::vineyard_array_type>
      local_oid_arrays_;
  // when oid_t is not string_view, only current fragment's is non-empty
  std::vector<std::vector<typename InternalType<oid_t>::vineyard_array_type>>
      oid_arrays_;

  std::vector<std::vector<vineyard::Hashmap<oid_t, vid_t>>> o2i_;

  // for non-string_view oid type
  std::vector<std::vector<vineyard::Hashmap<vid_t, oid_t>>> i2o_;
  // for string_view oid_type
  std::vector<std::vector<vineyard::Hashmap<vid_t, vid_t /* index */>>>
      i2o_index_;

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
