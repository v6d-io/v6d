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

#ifndef MODULES_GRAPH_LOADER_BASIC_ARROW_FRAGMENT_LOADER_H_
#define MODULES_GRAPH_LOADER_BASIC_ARROW_FRAGMENT_LOADER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/util/config.h"
#include "client/client.h"
#include "grape/worker/comm_spec.h"
#include "io/io/local_io_adaptor.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/partitioner.h"
#include "graph/utils/table_shuffler.h"
#include "graph/vertex_map/arrow_vertex_map.h"

namespace vineyard {
template <typename OID_T, typename VID_T, typename PARTITIONER_T>
class BasicArrowFragmentLoader {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using oid_array_t = typename vineyard::ConvertToArrowType<oid_t>::ArrayType;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using partitioner_t = PARTITIONER_T;

 public:
  constexpr static const char* ID_COLUMN = "id_column";
  constexpr static const char* SRC_COLUMN = "src_column";
  constexpr static const char* DST_COLUMN = "dst_column";

  explicit BasicArrowFragmentLoader(const grape::CommSpec& comm_spec)
      : comm_spec_(comm_spec) {}

  void SetPartitioner(const PARTITIONER_T& partitioner) {
    partitioner_ = partitioner;
  }

  void Init(const std::vector<std::shared_ptr<arrow::Table>>& vertex_tables,
            const std::vector<std::shared_ptr<arrow::Table>>& edge_tables) {
    vertex_tables_ = vertex_tables;
    edge_tables_ = edge_tables;
    v_label_num_ = vertex_tables.size();
    e_label_num_ = edge_tables.size();
    oid_lists_.resize(v_label_num_);
  }

  std::vector<std::vector<std::shared_ptr<oid_array_t>>>& GetOidLists() {
    return oid_lists_;
  }

  auto ShuffleVertexTables()
      -> boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>> {
    std::vector<std::shared_ptr<arrow::Table>> local_v_tables(v_label_num_);

    for (label_id_t v_label = 0; v_label < v_label_num_; v_label++) {
      auto e = boost::leaf::try_handle_all(
          [&, this]() -> boost::leaf::result<GSError> {
            auto& vertex_table = vertex_tables_[v_label];
            auto metadata = vertex_table->schema()->metadata();
            auto meta_idx = metadata->FindKey(ID_COLUMN);
            CHECK_OR_RAISE(meta_idx != -1);
            auto id_column_idx = std::stoi(metadata->value(meta_idx));
            // TODO(guanyi.gl): Failure occurred before MPI calling will make
            // processes hanging. We have to resolve this kind of issue.
            BOOST_LEAF_AUTO(tmp_table,
                            ShufflePropertyVertexTable<partitioner_t>(
                                comm_spec_, partitioner_, vertex_table));
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
            ARROW_OK_OR_RAISE(tmp_table->RemoveColumn(
                id_column_idx, &local_v_tables[v_label]));
#else
            ARROW_OK_ASSIGN_OR_RAISE(local_v_tables[v_label],
                                     tmp_table->RemoveColumn(id_column_idx));
#endif
            CHECK_OR_RAISE(tmp_table->field(id_column_idx)->type() ==
                           vineyard::ConvertToArrowType<oid_t>::TypeValue());
            CHECK_OR_RAISE(tmp_table->column(id_column_idx)->num_chunks() <= 1);
            auto local_oid_array = std::dynamic_pointer_cast<oid_array_t>(
                tmp_table->column(id_column_idx)->chunk(0));
            BOOST_LEAF_AUTO(
                r, FragmentAllGatherArray<oid_t>(comm_spec_, local_oid_array));
            oid_lists_[v_label] = r;

            return AllGatherError(comm_spec_);
          },
          [this](GSError& e) { return AllGatherError(e, comm_spec_); },
          [this](const boost::leaf::error_info& unmatched) {
            GSError e(ErrorCode::kIOError, "Unmatched error");
            return AllGatherError(e, comm_spec_);
          });
      if (e.error_code != ErrorCode::kOk) {
        return boost::leaf::new_error(e);
      }
    }

    return local_v_tables;
  }

  auto ShuffleEdgeTables(
      std::function<bool(fid_t, internal_oid_t, vid_t&)> mapper)
      -> boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>> {
    std::vector<std::shared_ptr<arrow::Table>> local_e_tables(e_label_num_);
    vineyard::IdParser<vid_t> id_parser;
    std::shared_ptr<arrow::Field> src_gid_field =
        std::make_shared<arrow::Field>(
            "src", vineyard::ConvertToArrowType<vid_t>::TypeValue());
    std::shared_ptr<arrow::Field> dst_gid_field =
        std::make_shared<arrow::Field>(
            "dst", vineyard::ConvertToArrowType<vid_t>::TypeValue());

    id_parser.Init(comm_spec_.fnum(), v_label_num_);

    for (label_id_t e_label = 0; e_label < e_label_num_; e_label++) {
      auto e = boost::leaf::try_handle_all(
          [&, this]() -> boost::leaf::result<GSError> {
            auto& edge_table = edge_tables_[e_label];
            auto metadata = edge_table->schema()->metadata();
            auto meta_idx_src = metadata->FindKey(SRC_COLUMN);
            auto meta_idx_dst = metadata->FindKey(DST_COLUMN);
            CHECK_OR_RAISE(meta_idx_src != -1);
            CHECK_OR_RAISE(meta_idx_dst != -1);
            auto src_column_idx = std::stoi(metadata->value(meta_idx_src));
            auto dst_column_idx = std::stoi(metadata->value(meta_idx_dst));
            BOOST_LEAF_AUTO(src_gid_array,
                            parseOidChunkedArray(
                                edge_table->column(src_column_idx), mapper));
            BOOST_LEAF_AUTO(dst_gid_array,
                            parseOidChunkedArray(
                                edge_table->column(dst_column_idx), mapper));

        // replace oid columns with gid
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
            ARROW_OK_OR_RAISE(edge_table->SetColumn(
                src_column_idx, src_gid_field, src_gid_array, &edge_table));
            ARROW_OK_OR_RAISE(edge_table->SetColumn(
                dst_column_idx, dst_gid_field, dst_gid_array, &edge_table));
#else
            ARROW_OK_ASSIGN_OR_RAISE(
                edge_table, edge_table->SetColumn(src_column_idx, src_gid_field,
                                                  src_gid_array));
            ARROW_OK_ASSIGN_OR_RAISE(
                edge_table, edge_table->SetColumn(dst_column_idx, dst_gid_field,
                                                  dst_gid_array));
#endif

            auto r = ShufflePropertyEdgeTable<vid_t>(
                comm_spec_, id_parser, src_column_idx, dst_column_idx,
                edge_table);
            BOOST_LEAF_CHECK(r);
            local_e_tables[e_label] = r.value();
            return AllGatherError(comm_spec_);
          },
          [this](GSError& e) { return AllGatherError(e, comm_spec_); },
          [this](const boost::leaf::error_info& unmatched) {
            GSError e(ErrorCode::kIOError, "Unmatched error");
            return AllGatherError(e, comm_spec_);
          });
      if (e.error_code != ErrorCode::kOk) {
        return boost::leaf::new_error(e);
      }
    }

    return local_e_tables;
  }

 private:
  auto parseOidChunkedArray(
      const std::shared_ptr<arrow::ChunkedArray>& oid_arrays_in,
      std::function<bool(fid_t, internal_oid_t, vid_t&)>& oid2gid_mapper)
      -> boost::leaf::result<std::shared_ptr<arrow::ChunkedArray>> {
    size_t chunk_num = oid_arrays_in->num_chunks();
    std::vector<std::shared_ptr<arrow::Array>> chunks_out(chunk_num);

    for (size_t chunk_i = 0; chunk_i != chunk_num; ++chunk_i) {
      std::shared_ptr<oid_array_t> oid_array =
          std::dynamic_pointer_cast<oid_array_t>(oid_arrays_in->chunk(chunk_i));
      typename vineyard::ConvertToArrowType<vid_t>::BuilderType builder;
      size_t size = oid_array->length();
      ARROW_OK_OR_RAISE(builder.Resize(size));

      for (size_t i = 0; i != size; ++i) {
        internal_oid_t oid = oid_array->GetView(i);
        fid_t fid = partitioner_.GetPartitionId(oid_t(oid));
        CHECK_OR_RAISE(oid2gid_mapper(fid, oid, builder[i]));
      }
      ARROW_OK_OR_RAISE(builder.Advance(size));
      ARROW_OK_OR_RAISE(builder.Finish(&chunks_out[chunk_i]));
    }

    return std::make_shared<arrow::ChunkedArray>(chunks_out);
  }

  grape::CommSpec comm_spec_;
  label_id_t v_label_num_;
  label_id_t e_label_num_;
  std::vector<std::shared_ptr<arrow::Table>> vertex_tables_;
  std::vector<std::shared_ptr<arrow::Table>> edge_tables_;

  std::vector<std::vector<std::shared_ptr<oid_array_t>>>
      oid_lists_;  // v_label/fid/oid_array

  partitioner_t partitioner_;
};
}  // namespace vineyard
#endif  // MODULES_GRAPH_LOADER_BASIC_ARROW_FRAGMENT_LOADER_H_
