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

#ifndef MODULES_GRAPH_VERTEX_MAP_ARROW_VERTEX_MAP_H_
#define MODULES_GRAPH_VERTEX_MAP_ARROW_VERTEX_MAP_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "basic/ds/arrow.h"
#include "basic/ds/hashmap.h"
#include "client/client.h"
#include "common/util/typename.h"

#include "graph/fragment/property_graph_types.h"
#include "graph/vertex_map/arrow_bisector.h"

namespace vineyard {

template <typename OID_T, typename VID_T>
class ArrowVertexMapBuilder;

template <typename OID_T, typename VID_T>
class BasicArrowVertexMapBuilder;

template <typename OID_T, typename VID_T>
class ArrowVertexMap
    : public vineyard::Registered<ArrowVertexMap<OID_T, VID_T>> {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using oid_array_t = ArrowArrayType<oid_t>;
  using vid_array_t = ArrowArrayType<vid_t>;

  static_assert(!std::is_same<OID_T, std::string>::value,
                "Expect arrow_string_view in vertex map's OID_T");

 public:
  ArrowVertexMap() {}
  ~ArrowVertexMap() {}

  static std::unique_ptr<vineyard::Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<vineyard::Object>(
        std::unique_ptr<ArrowVertexMap<OID_T, VID_T>>{
            new ArrowVertexMap<OID_T, VID_T>()});
  }

  void Construct(const vineyard::ObjectMeta& meta);

  bool GetOid(vid_t gid, oid_t& oid) const;

  std::shared_ptr<oid_array_t> GetOids(fid_t fid, label_id_t label_id) const;

  std::shared_ptr<oid_array_t> GetOidArray(fid_t fid, label_id_t label_id);

  bool GetGid(fid_t fid, label_id_t label_id, oid_t oid, vid_t& gid) const;

  bool GetGid(label_id_t label_id, oid_t oid, vid_t& gid) const;

  fid_t fnum() const { return fnum_; }

  size_t GetTotalNodesNum() const;

  size_t GetTotalNodesNum(label_id_t label) const;

  label_id_t label_num() const { return label_num_; }

  VID_T GetInnerVertexSize(fid_t fid) const;

  VID_T GetInnerVertexSize(fid_t fid, label_id_t label_id) const;

  ObjectID AddVertices(
      Client& client,
      std::map<label_id_t, std::vector<std::shared_ptr<oid_array_t>>>
          oid_arrays_map);

  ObjectID AddVertices(
      Client& client,
      std::map<label_id_t, std::vector<std::shared_ptr<arrow::ChunkedArray>>>
          oid_arrays_map);

  ObjectID AddNewVertexLabels(
      Client& client,
      // label index -> fid -> ..
      std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_arrays);

  ObjectID AddNewVertexLabels(
      Client& client,
      // label index -> fid -> ..
      std::vector<std::vector<std::shared_ptr<arrow::ChunkedArray>>>
          oid_arrays);

 private:
  ObjectID addNewVertexLabels(
      Client& client,
      // label index -> fid -> ..
      std::vector<std::vector<std::vector<std::shared_ptr<oid_array_t>>>>
          oid_arrays);

  fid_t fnum_;
  label_id_t label_num_;
  bool bisect_ = false;

  vineyard::IdParser<vid_t> id_parser_;

  // frag->label->oid
  std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_arrays_;

  // bisect
  std::vector<std::vector<std::shared_ptr<arrow::UInt64Array>>>
      oid_array_indices_;
  std::vector<std::vector<ArrowBisector<oid_t>>> o2g_bisector_;
  // flat hashmap
  std::vector<std::vector<vineyard::Hashmap<oid_t, vid_t>>> o2g_;

  friend class ArrowVertexMapBuilder<OID_T, VID_T>;
  friend class BasicArrowVertexMapBuilder<OID_T, VID_T>;
};

template <typename OID_T, typename VID_T>
class ArrowVertexMapBuilder : public vineyard::ObjectBuilder {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using vineyard_internal_oid_array_t =
      typename InternalType<oid_t>::vineyard_array_type;

  static_assert(!std::is_same<OID_T, std::string>::value,
                "Expect arrow_string_view in local vertex map's OID_T");

 public:
  explicit ArrowVertexMapBuilder(vineyard::Client& client) {}

  void set_fnum_label_num(const fid_t fnum, const label_id_t label_num);

  void set_bisect(const bool bisect = false);

  void set_oid_array(fid_t fid, label_id_t label,
                     const vineyard_internal_oid_array_t& array);

  void set_oid_array(
      fid_t fid, label_id_t label,
      const std::shared_ptr<vineyard_internal_oid_array_t>& array);

  void set_oid_array_indices(fid_t fid, label_id_t label,
                             const UInt64Array& array);

  void set_oid_array_indices(fid_t fid, label_id_t label,
                             const std::shared_ptr<UInt64Array>& array);

  void set_o2g(fid_t fid, label_id_t label,
               const vineyard::Hashmap<oid_t, vid_t>& rm);

  void set_o2g(fid_t fid, label_id_t label,
               const std::shared_ptr<vineyard::Hashmap<oid_t, vid_t>>& rm);

  std::shared_ptr<vineyard::Object> _Seal(vineyard::Client& client);

 private:
  fid_t fnum_;
  label_id_t label_num_;
  bool bisect_ = false;

  std::vector<std::vector<vineyard_internal_oid_array_t>> oid_arrays_;
  std::vector<std::vector<UInt64Array>> oid_array_indices_;
  std::vector<std::vector<Hashmap<oid_t, vid_t>>> o2g_;
};

template <typename OID_T, typename VID_T>
class BasicArrowVertexMapBuilder : public ArrowVertexMapBuilder<OID_T, VID_T> {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using oid_array_t = ArrowArrayType<oid_t>;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;

  static_assert(!std::is_same<OID_T, std::string>::value,
                "Expect arrow_string_view in local vertex map's OID_T");

 public:
  BasicArrowVertexMapBuilder(
      vineyard::Client& client, fid_t fnum, label_id_t label_num,
      std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_arrays);

  BasicArrowVertexMapBuilder(
      vineyard::Client& client, fid_t fnum, label_id_t label_num, bool bisect,
      std::vector<std::vector<std::shared_ptr<oid_array_t>>> oid_arrays);

  BasicArrowVertexMapBuilder(
      vineyard::Client& client, fid_t fnum, label_id_t label_num,
      std::vector<std::vector<std::shared_ptr<arrow::ChunkedArray>>>
          oid_arrays);

  BasicArrowVertexMapBuilder(
      vineyard::Client& client, fid_t fnum, label_id_t label_num, bool bisect,
      std::vector<std::vector<std::shared_ptr<arrow::ChunkedArray>>>
          oid_arrays);

  vineyard::Status Build(vineyard::Client& client) override;

 private:
  fid_t fnum_;
  label_id_t label_num_;
  bool bisect_ = false;

  vineyard::IdParser<vid_t> id_parser_;

  std::vector<std::vector<std::vector<std::shared_ptr<oid_array_t>>>>
      oid_arrays_;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_VERTEX_MAP_ARROW_VERTEX_MAP_H_
