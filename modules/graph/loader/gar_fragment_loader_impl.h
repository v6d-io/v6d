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

#ifndef MODULES_GRAPH_LOADER_GAR_FRAGMENT_LOADER_IMPL_H_
#define MODULES_GRAPH_LOADER_GAR_FRAGMENT_LOADER_IMPL_H_

#ifdef ENABLE_GAR

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gar/reader/arrow_chunk_reader.h"
#include "gar/utils/general_params.h"

#include "graph/fragment/property_graph_utils.h"
#include "graph/fragment/property_graph_utils_impl.h"
#include "graph/loader/fragment_loader_utils.h"
#include "graph/loader/gar_fragment_loader.h"

namespace vineyard {

#ifndef RETURN_GS_ERROR_IF_NOT_OK
#define RETURN_GS_ERROR_IF_NOT_OK(status)                          \
  do {                                                             \
    if (!status.ok()) {                                            \
      RETURN_GS_ERROR(ErrorCode::kGraphArError, status.message()); \
    }                                                              \
  } while (false);
#endif

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::GARFragmentLoader(
    Client& client, const grape::CommSpec& comm_spec,
    const std::string& graph_info_yaml, bool directed, bool generate_eid)
    : client_(client),
      comm_spec_(comm_spec),
      directed_(directed),
      generate_eid_(generate_eid) {
  // Load graph info.
  auto maybe_graph_info = GraphArchive::GraphInfo::Load(graph_info_yaml);
  if (!maybe_graph_info.status().ok()) {
    LOG(ERROR) << "Failed to load graph info from " << graph_info_yaml
               << ", error: " << maybe_graph_info.status().message();
  }
  graph_info_ = std::make_shared<GraphArchive::GraphInfo>(
      std::move(maybe_graph_info.value()));
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<ObjectID>
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::LoadFragment() {
  // distribute the vertices for fragments.
  BOOST_LEAF_CHECK(distributeVertices());
  // Load vertex tables.
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "LOADING-VERTEX-TABLES-0";
  BOOST_LEAF_CHECK(LoadVertexTables());
  LOG_IF(INFO, !comm_spec_.worker_id())
      << MARKER << "LOADING-VERTEX-TABLES-100";
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after loading vertex tables: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();
  // Construct vertex map.
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-VERTEX-MAP-0";
  BOOST_LEAF_CHECK(constructVertexMap());
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-VERTEX-MAP-100";
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after construct vertex map: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();
  // Load edge tables.
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "LOADING-EDGE-TABLES-0";
  BOOST_LEAF_CHECK(LoadEdgeTables());
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "LOADING-EDGE-TABLES-100";
  VLOG(100) << "[worker-" << comm_spec_.worker_id()
            << "] RSS after loading edge tables: " << get_rss_pretty()
            << ", peak = " << get_peak_rss_pretty();

  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "CONSTRUCT-FRAGMENT-0";
  return this->ConstructFragment();
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<ObjectID>
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::LoadFragmentAsFragmentGroup() {
  BOOST_LEAF_AUTO(frag_id, LoadFragment());
  auto frag =
      std::dynamic_pointer_cast<ArrowFragment<OID_T, VID_T, vertex_map_t>>(
          client_.GetObject(frag_id));
  if (frag == nullptr) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                    "fragment is null, means it is failed to be constructed");
  }

  BOOST_LEAF_AUTO(group_id,
                  ConstructFragmentGroup(client_, frag_id, comm_spec_));
  return group_id;
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<void>
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::LoadVertexTables() {
  vertex_tables_.resize(vertex_label_num_);
  for (const auto& vertex_label : vertex_labels_) {
    BOOST_LEAF_CHECK(loadVertexTableOfLabel(vertex_label));
  }
  return {};
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<void>
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::LoadEdgeTables() {
  csr_edge_tables_with_label_.resize(edge_label_num_);
  csc_edge_tables_with_label_.resize(edge_label_num_);
  // read the adj list chunk tables
  for (const auto& edge : graph_info_->GetEdgeInfos()) {
    const auto& edge_info = edge.second;
    if (edge_info.ContainAdjList(
            GraphArchive::AdjListType::ordered_by_source)) {
      loadEdgeTableOfLabel(edge_info,
                           GraphArchive::AdjListType::ordered_by_source);
    }
    if (this->directed_) {
      if (edge_info.ContainAdjList(
              GraphArchive::AdjListType::ordered_by_dest)) {
        loadEdgeTableOfLabel(edge_info,
                             GraphArchive::AdjListType::ordered_by_dest);
      }
    }
  }
  for (size_t i = 0; i < csr_edge_tables_with_label_.size(); ++i) {
    csr_edge_tables_.push_back(
        FlattenTableInfos(csr_edge_tables_with_label_[i]));
  }
  if (this->directed_) {
    for (size_t i = 0; i < csc_edge_tables_with_label_.size(); ++i) {
      csc_edge_tables_.push_back(
          FlattenTableInfos(csc_edge_tables_with_label_[i]));
    }
  }
  csr_edge_tables_with_label_.clear();
  csc_edge_tables_with_label_.clear();
  edge_label_num_ = edge_labels_.size();
  return {};
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<ObjectID>
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::ConstructFragment() {
  GARFragmentBuilder<oid_t, vid_t, vertex_map_t> frag_builder(client_, vm_ptr_);

  PropertyGraphSchema schema;
  BOOST_LEAF_CHECK(initSchema(schema));
  frag_builder.SetPropertyGraphSchema(std::move(schema));

  int thread_num =
      (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
      comm_spec_.local_num();

  BOOST_LEAF_CHECK(
      frag_builder.Init(comm_spec_.fid(), comm_spec_.fnum(),
                        std::move(vertex_tables_), std::move(csr_edge_tables_),
                        std::move(csc_edge_tables_), directed_, thread_num));

  std::shared_ptr<Object> fragment_object;
  VY_OK_OR_RAISE(frag_builder.Seal(client_, fragment_object));
  auto frag = std::dynamic_pointer_cast<fragment_t>(fragment_object);

  VY_OK_OR_RAISE(client_.Persist(frag->id()));
  return frag->id();
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<void>
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::distributeVertices() {
  for (const auto& item : graph_info_->GetVertexInfos()) {
    const auto& label = item.first;
    const auto& vertex_info = item.second;
    if (std::find(vertex_labels_.begin(), vertex_labels_.end(), label) !=
        vertex_labels_.end()) {
      continue;
    }
    vertex_labels_.push_back(label);
    vertex_chunk_sizes_.push_back(vertex_info.GetChunkSize());
    auto chunk_num = GraphArchive::utils::GetVertexChunkNum(
        graph_info_->GetPrefix(), vertex_info);
    RETURN_GS_ERROR_IF_NOT_OK(chunk_num.status());
    // distribute the vertex chunks for fragments
    int64_t bsize = chunk_num.value() / static_cast<int64_t>(comm_spec_.fnum());
    vertex_chunk_begin_of_frag_[label].resize(comm_spec_.fnum() + 1, 0);
    for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
      vertex_chunk_begin_of_frag_[label][fid] =
          static_cast<gar_id_t>(fid) * bsize;
    }
    vertex_chunk_begin_of_frag_[label][comm_spec_.fnum()] = chunk_num.value();
  }
  vertex_label_num_ = vertex_labels_.size();
  for (size_t i = 0; i < vertex_labels_.size(); ++i) {
    vertex_label_to_index_[vertex_labels_[i]] = i;
  }

  for (const auto& item : graph_info_->GetEdgeInfos()) {
    auto& edge_info = item.second;
    // record edge label
    const auto& edge_label = edge_info.GetEdgeLabel();
    const auto& src_label = edge_info.GetSrcLabel();
    const auto& dst_label = edge_info.GetDstLabel();
    auto it = std::find(edge_labels_.begin(), edge_labels_.end(), edge_label);
    if (it == edge_labels_.end()) {
      edge_labels_.push_back(edge_label);
      edge_label_to_index_[edge_label] = edge_labels_.size() - 1;
      edge_relations_.resize(edge_labels_.size());
    }
    edge_relations_[edge_label_to_index_[edge_label]].emplace(
        vertex_label_to_index_[src_label], vertex_label_to_index_[dst_label]);
  }
  edge_label_num_ = edge_labels_.size();

  vid_parser_.Init(comm_spec_.fnum(), this->vertex_label_num_);
  return {};
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<void>
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::constructVertexMap() {
  std::vector<std::vector<std::shared_ptr<arrow::ChunkedArray>>> oid_lists(
      vertex_label_num_);
  ThreadGroup tg(comm_spec_);
  auto shuffle_procedure = [&](const label_id_t label_id) -> Status {
    std::vector<std::shared_ptr<arrow::ChunkedArray>> shuffled_oid_array;
    auto& vertex_info =
        graph_info_->GetVertexInfo(vertex_labels_[label_id]).value();
    const auto& property_groups = vertex_info.GetPropertyGroups();
    std::string primary_key;
    for (const auto& pg : property_groups) {
      for (const auto& prop : pg.GetProperties()) {
        if (prop.is_primary) {
          primary_key = prop.name;
          break;
        }
      }
      if (!primary_key.empty()) {
        break;
      }
    }
    if (primary_key.empty()) {
      std::string msg = "primary key is not found in " +
                        vertex_labels_[label_id] + " property groups";
      return Status::Invalid(msg);
    }
    auto local_oid_array =
        vertex_tables_[label_id]->GetColumnByName(primary_key);
    if (local_oid_array == nullptr) {
      std::string msg = "primary key column " + primary_key +
                        " is not found in " + vertex_labels_[label_id] +
                        " table";
      return Status::Invalid(msg);
    }
    RETURN_ON_ERROR(FragmentAllGatherArray(comm_spec_, local_oid_array,
                                           shuffled_oid_array));
    for (auto const& array : shuffled_oid_array) {
      oid_lists[label_id].emplace_back(
          std::dynamic_pointer_cast<arrow::ChunkedArray>(array));
    }
    return Status::OK();
  };
  for (label_id_t label_id = 0; label_id < vertex_label_num_; ++label_id) {
    tg.AddTask(shuffle_procedure, label_id);
  }
  {
    Status status;
    for (auto const& s : tg.TakeResults()) {
      status += s;
    }
    VY_OK_OR_RAISE(status);
  }

  BasicArrowVertexMapBuilder<internal_oid_t, vid_t> vm_builder(
      client_, comm_spec_.fnum(), vertex_label_num_, std::move(oid_lists));
  std::shared_ptr<Object> vm_object;
  VY_OK_OR_RAISE(vm_builder.Seal(client_, vm_object));
  vm_ptr_ = std::dynamic_pointer_cast<vertex_map_t>(vm_object);
  return {};
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<void>
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::loadVertexTableOfLabel(
    const std::string& vertex_label) {
  auto maybe_vertex_info = graph_info_->GetVertexInfo(vertex_label);
  RETURN_GS_ERROR_IF_NOT_OK(maybe_vertex_info.status());
  auto& vertex_info = maybe_vertex_info.value();
  const auto& label = vertex_info.GetLabel();
  label_id_t label_id = vertex_label_to_index_[label];
  auto vertex_chunk_begin =
      vertex_chunk_begin_of_frag_[label][comm_spec_.fid()];
  auto vertex_chunk_num_of_fragment =
      vertex_chunk_begin_of_frag_[label][comm_spec_.fid() + 1] -
      vertex_chunk_begin_of_frag_[label][comm_spec_.fid()];
  auto chunk_size = vertex_info.GetChunkSize();
  const auto& property_groups = vertex_info.GetPropertyGroups();

  table_vec_t pg_tables;
  int64_t thread_num =
      (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
      comm_spec_.local_num();
  std::vector<std::thread> threads(thread_num);
  int64_t batch_size =
      (vertex_chunk_num_of_fragment + thread_num - 1) / thread_num;
  for (const auto& pg : property_groups) {
    table_vec_t vertex_chunk_tables(vertex_chunk_num_of_fragment);
    std::atomic<int64_t> cur_chunk_index(0);
    for (int64_t i = 0; i < thread_num; ++i) {
      threads[i] = std::thread([&]() -> boost::leaf::result<void> {
        auto maybe_reader =
            GraphArchive::ConstructVertexPropertyArrowChunkReader(
                *(graph_info_.get()), label, pg);
        RETURN_GS_ERROR_IF_NOT_OK(maybe_reader.status());
        auto& reader = maybe_reader.value();
        while (true) {
          int64_t begin = cur_chunk_index.fetch_add(batch_size);
          if (begin >= vertex_chunk_num_of_fragment) {
            break;
          }
          int64_t end = std::min(static_cast<int64_t>(begin + batch_size),
                                 vertex_chunk_num_of_fragment);
          int64_t iter = begin;
          while (iter != end) {
            RETURN_GS_ERROR_IF_NOT_OK(
                reader.seek((vertex_chunk_begin + iter) * chunk_size));
            auto chunk_table = reader.GetChunk();
            RETURN_GS_ERROR_IF_NOT_OK(chunk_table.status());
            vertex_chunk_tables[iter] = chunk_table.value();
            ++iter;
          }
        }
        return {};
      });
    }
    for (auto& t : threads) {
      t.join();
    }
    auto pg_table = arrow::ConcatenateTables(vertex_chunk_tables);
    if (!pg_table.status().ok()) {
      RETURN_GS_ERROR(ErrorCode::kArrowError, pg_table.status().message());
    }
    pg_tables.push_back(std::move(pg_table).ValueOrDie());
  }
  std::shared_ptr<arrow::Table> concat_table;
  VY_OK_OR_RAISE(ConcatenateTablesColumnWise(pg_tables, concat_table));
  // loosen the data type
  std::shared_ptr<arrow::Schema> normalized_schema;
  VY_OK_OR_RAISE(TypeLoosen({concat_table->schema()}, normalized_schema));
  std::shared_ptr<arrow::Table> table_out;
  VY_OK_OR_RAISE(CastTableToSchema(concat_table, normalized_schema, table_out));
  auto metadata = std::make_shared<arrow::KeyValueMetadata>();
  metadata->Append("label", label);
  metadata->Append("label_id", std::to_string(label_id));
  metadata->Append("type", PropertyGraphSchema::VERTEX_TYPE_NAME);
  metadata->Append("retain_oid", std::to_string(false));
  vertex_tables_[label_id] = table_out->ReplaceSchemaMetadata(metadata);
  return {};
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<void>
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::loadEdgeTableOfLabel(
    const GraphArchive::EdgeInfo& edge_info,
    GraphArchive::AdjListType adj_list_type) {
  auto src_label = edge_info.GetSrcLabel();
  auto dst_label = edge_info.GetDstLabel();
  auto edge_label = edge_info.GetEdgeLabel();
  const auto& property_groups =
      edge_info.GetPropertyGroups(adj_list_type).value();

  int64_t vertex_chunk_begin = 0;
  int64_t vertex_chunk_num_of_fragment = 0;
  int64_t edge_chunk_size = edge_info.GetChunkSize();
  int64_t vertex_chunk_size;
  if (adj_list_type == GraphArchive::AdjListType::ordered_by_source) {
    vertex_chunk_begin =
        vertex_chunk_begin_of_frag_[src_label][comm_spec_.fid()];
    vertex_chunk_num_of_fragment =
        vertex_chunk_begin_of_frag_[src_label][comm_spec_.fid() + 1] -
        vertex_chunk_begin_of_frag_[src_label][comm_spec_.fid()];
    vertex_chunk_size = edge_info.GetSrcChunkSize();
  } else {
    vertex_chunk_begin =
        vertex_chunk_begin_of_frag_[dst_label][comm_spec_.fid()];
    vertex_chunk_num_of_fragment =
        vertex_chunk_begin_of_frag_[dst_label][comm_spec_.fid() + 1] -
        vertex_chunk_begin_of_frag_[dst_label][comm_spec_.fid()];
    vertex_chunk_size = edge_info.GetDstChunkSize();
  }
  std::vector<std::shared_ptr<arrow::Int64Array>> offset_arrays(
      vertex_chunk_num_of_fragment);

  int64_t thread_num =
      (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
      comm_spec_.local_num();
  std::vector<std::thread> threads(thread_num);
  std::vector<GraphArchive::IdType> agg_edge_chunk_num(
      vertex_chunk_num_of_fragment + 1, 0);
  int64_t batch_size =
      (vertex_chunk_num_of_fragment + thread_num - 1) / thread_num;
  std::atomic<int64_t> cur_chunk(0);
  // read the offset arrays
  for (int64_t i = 0; i < thread_num; ++i) {
    threads[i] = std::thread([&]() -> boost::leaf::result<void> {
      auto maybe_offset_reader =
          GraphArchive::ConstructAdjListOffsetArrowChunkReader(
              *(graph_info_.get()), src_label, edge_label, dst_label,
              adj_list_type);
      RETURN_GS_ERROR_IF_NOT_OK(maybe_offset_reader.status());
      auto& offset_reader = maybe_offset_reader.value();
      while (true) {
        int64_t begin = cur_chunk.fetch_add(batch_size);
        if (begin >= vertex_chunk_num_of_fragment) {
          break;
        }
        int64_t end = std::min(static_cast<int64_t>(begin + batch_size),
                               vertex_chunk_num_of_fragment);
        int64_t iter = begin;
        while (iter != end) {
          int64_t vertex_chunk_id = iter + vertex_chunk_begin;
          RETURN_GS_ERROR_IF_NOT_OK(
              offset_reader.seek(vertex_chunk_id * vertex_chunk_size));
          auto offset_result = offset_reader.GetChunk();
          RETURN_GS_ERROR_IF_NOT_OK(offset_result.status());
          offset_arrays[iter] = std::dynamic_pointer_cast<arrow::Int64Array>(
              offset_result.value());

          // get edge num of this vertex chunk from offset array
          int64_t edge_num =
              offset_arrays[iter]->GetView(offset_arrays[iter]->length() - 1);
          agg_edge_chunk_num[iter] =
              (edge_num + edge_chunk_size - 1) / edge_chunk_size;
          ++iter;
        }
      }
      return {};
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  // prefix sum the offset arrays to a whole offset array
  std::shared_ptr<arrow::Int64Array> offset_array;
  VY_OK_OR_RAISE(parallel_prefix_sum_chunks(offset_arrays, offset_array));

  for (size_t i = 1; i < agg_edge_chunk_num.size() - 1; ++i) {
    agg_edge_chunk_num[i] += agg_edge_chunk_num[i - 1];
  }
  for (size_t i = agg_edge_chunk_num.size() - 1; i > 0; --i) {
    agg_edge_chunk_num[i] = agg_edge_chunk_num[i - 1];
  }
  agg_edge_chunk_num[0] = 0;

  auto total_edge_chunk_num = agg_edge_chunk_num.back();
  table_vec_t edge_chunk_tables(total_edge_chunk_num);
  std::vector<table_vec_t> edge_property_chunk_tables(property_groups.size());
  for (size_t i = 0; i < property_groups.size(); ++i) {
    edge_property_chunk_tables[i].resize(total_edge_chunk_num);
  }
  std::atomic<int64_t> cur(0);
  batch_size = (total_edge_chunk_num + thread_num - 1) / thread_num;
  for (int64_t i = 0; i < thread_num; ++i) {
    threads[i] = std::thread([&]() -> boost::leaf::result<void> {
      auto expect = GraphArchive::ConstructAdjListArrowChunkReader(
          *(graph_info_.get()), src_label, edge_label, dst_label,
          adj_list_type);
      RETURN_GS_ERROR_IF_NOT_OK(expect.status());
      auto& reader = expect.value();
      std::vector<GraphArchive::AdjListPropertyArrowChunkReader>
          property_readers;
      for (const auto& pg : property_groups) {
        property_readers.emplace_back(edge_info, pg, adj_list_type,
                                      graph_info_->GetPrefix());
      }
      while (true) {
        int64_t begin = cur.fetch_add(batch_size);
        if (begin >= total_edge_chunk_num) {
          break;
        }
        int64_t end = std::min(static_cast<int64_t>(begin + batch_size),
                               total_edge_chunk_num);
        int64_t iter = begin;
        while (iter != end) {
          // get the vertex_chunk_index & edge_chunk_index pair from global edge
          // chunk id
          auto chunk_pair = BinarySearchChunkPair(agg_edge_chunk_num, iter);
          auto vertex_chunk_id = chunk_pair.first + vertex_chunk_begin;
          auto edge_chunk_index = chunk_pair.second;
          RETURN_GS_ERROR_IF_NOT_OK(
              reader.seek_chunk_index(vertex_chunk_id, edge_chunk_index));
          auto edge_chunk_result = reader.GetChunk();
          RETURN_GS_ERROR_IF_NOT_OK(edge_chunk_result.status());
          edge_chunk_tables[iter] = edge_chunk_result.value();
          for (size_t j = 0; j < property_groups.size(); ++j) {
            auto& pg_reader = property_readers[j];
            RETURN_GS_ERROR_IF_NOT_OK(
                pg_reader.seek_chunk_index(vertex_chunk_id, edge_chunk_index));
            auto pg_result = pg_reader.GetChunk();
            RETURN_GS_ERROR_IF_NOT_OK(pg_result.status());
            edge_property_chunk_tables[j][iter] = pg_result.value();
          }
          ++iter;
        }
      }
      return {};
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  // process adj list tables
  auto adj_list_table = arrow::ConcatenateTables(edge_chunk_tables);
  RETURN_GS_ERROR_IF_NOT_OK(adj_list_table.status());
  std::shared_ptr<arrow::Table> adj_list_table_with_gid;
  BOOST_LEAF_ASSIGN(
      adj_list_table_with_gid,
      parseEdgeIdArrays(std::move(adj_list_table).ValueOrDie(),
                        vertex_label_to_index_[src_label],
                        vertex_label_to_index_[dst_label], adj_list_type));

  // process property tables
  std::shared_ptr<arrow::Table> property_table;
  std::vector<std::shared_ptr<arrow::Table>> property_table_of_groups(
      property_groups.size());
  for (size_t i = 0; i < property_groups.size(); ++i) {
    auto property_chunk_table =
        arrow::ConcatenateTables(edge_property_chunk_tables[i]);
    RETURN_GS_ERROR_IF_NOT_OK(property_chunk_table.status());
    property_table_of_groups[i] = std::move(property_chunk_table).ValueOrDie();
  }

  label_id_t label_id = edge_label_to_index_[edge_label];
  auto metadata = std::make_shared<arrow::KeyValueMetadata>();
  metadata->Append("label", edge_label);
  metadata->Append("label_id", std::to_string(label_id));
  metadata->Append("type", PropertyGraphSchema::EDGE_TYPE_NAME);
  std::shared_ptr<arrow::Table> concat_property_table;
  if (!property_groups.empty()) {
    VY_OK_OR_RAISE(ConcatenateTablesColumnWise(property_table_of_groups,
                                               concat_property_table));
    // loosen the data type
    std::shared_ptr<arrow::Schema> normalized_schema;
    VY_OK_OR_RAISE(
        TypeLoosen({concat_property_table->schema()}, normalized_schema));
    VY_OK_OR_RAISE(CastTableToSchema(concat_property_table, normalized_schema,
                                     concat_property_table));
  } else {
    // create an empty arrow table
    concat_property_table =
        arrow::Table::MakeEmpty(arrow::schema({})).ValueOrDie();
  }
  property_table = concat_property_table->ReplaceSchemaMetadata(metadata);
  label_id_t source_label_id = vertex_label_to_index_[src_label];
  label_id_t destination_label_id = vertex_label_to_index_[dst_label];
  if (adj_list_type == GraphArchive::AdjListType::ordered_by_source) {
    csr_edge_tables_with_label_[label_id].emplace_back(
        adj_list_table_with_gid, offset_array, property_table, source_label_id,
        false);
  } else if (adj_list_type == GraphArchive::AdjListType::ordered_by_dest) {
    csc_edge_tables_with_label_[label_id].emplace_back(
        adj_list_table_with_gid, offset_array, property_table,
        destination_label_id, false);
  }
  return {};
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<void>
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::initSchema(
    PropertyGraphSchema& schema) {
  schema.set_fnum(comm_spec_.fnum());
  for (label_id_t v_label = 0; v_label != vertex_label_num_; ++v_label) {
    std::string vertex_label = vertex_labels_[v_label];
    auto entry =
        schema.CreateEntry(vertex_label, PropertyGraphSchema::VERTEX_TYPE_NAME);

    auto table = vertex_tables_[v_label];

    for (int i = 0; i < table->num_columns(); ++i) {
      entry->AddProperty(table->schema()->field(i)->name(),
                         table->schema()->field(i)->type());
    }
  }
  for (label_id_t e_label = 0; e_label != edge_label_num_; ++e_label) {
    std::string edge_label = edge_labels_[e_label];
    auto entry =
        schema.CreateEntry(edge_label, PropertyGraphSchema::EDGE_TYPE_NAME);

    auto& relation_set = edge_relations_[e_label];
    for (auto& pair : relation_set) {
      std::string src_label = vertex_labels_[pair.first];
      std::string dst_label = vertex_labels_[pair.second];
      entry->AddRelation(src_label, dst_label);
    }

    auto table = csr_edge_tables_[e_label].property_table;
    for (int i = 0; i < table->num_columns(); ++i) {
      entry->AddProperty(table->schema()->field(i)->name(),
                         table->schema()->field(i)->type());
    }
  }
  std::string message;
  if (!schema.Validate(message)) {
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError, message);
  }
  return {};
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
boost::leaf::result<std::shared_ptr<arrow::Table>>
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::parseEdgeIdArrays(
    std::shared_ptr<arrow::Table> adj_list_table, label_id_t src_label,
    label_id_t dst_label, GraphArchive::AdjListType adj_list_type) {
  std::shared_ptr<arrow::Field> src_gid_field = std::make_shared<arrow::Field>(
      GraphArchive::GeneralParams::kSrcIndexCol,
      vineyard::ConvertToArrowType<vid_t>::TypeValue());
  std::shared_ptr<arrow::ChunkedArray> src_gid_array;
  bool all_be_local_in_src = false, all_be_local_in_dst = false;
  if (adj_list_type == GraphArchive::AdjListType::ordered_by_source) {
    all_be_local_in_src = true;
    all_be_local_in_dst = false;
  } else if (adj_list_type == GraphArchive::AdjListType::ordered_by_dest) {
    all_be_local_in_src = false;
    all_be_local_in_dst = true;
  }
  auto src_id_array = adj_list_table->GetColumnByName(
      GraphArchive::GeneralParams::kSrcIndexCol);
  VY_OK_OR_RAISE(parseIdChunkedArray(src_label, src_id_array,
                                     all_be_local_in_src, src_gid_array));
  std::shared_ptr<arrow::Field> dst_gid_field = std::make_shared<arrow::Field>(
      GraphArchive::GeneralParams::kDstIndexCol,
      vineyard::ConvertToArrowType<vid_t>::TypeValue());
  std::shared_ptr<arrow::ChunkedArray> dst_gid_array;
  auto dst_id_array = adj_list_table->GetColumnByName(
      GraphArchive::GeneralParams::kDstIndexCol);
  VY_OK_OR_RAISE(parseIdChunkedArray(dst_label, dst_id_array,
                                     all_be_local_in_dst, dst_gid_array));

  // replace id columns with gid
  ARROW_OK_ASSIGN_OR_RAISE(
      adj_list_table,
      adj_list_table->SetColumn(0, src_gid_field, src_gid_array));
  ARROW_OK_ASSIGN_OR_RAISE(
      adj_list_table,
      adj_list_table->SetColumn(1, dst_gid_field, dst_gid_array));
  return adj_list_table;
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
vineyard::Status
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::parseIdChunkedArray(
    label_id_t label_id,
    const std::shared_ptr<arrow::ChunkedArray> id_arrays_in,
    bool all_be_local_vertex, std::shared_ptr<arrow::ChunkedArray>& out) {
  size_t chunk_num = id_arrays_in->num_chunks();
  std::vector<std::shared_ptr<arrow::Array>> chunks_out(chunk_num);

  auto parsefn = [&](const size_t chunk_index,
                     const std::shared_ptr<arrow::Array> oid_array) -> Status {
    return parseIdChunkedArrayChunk(label_id, oid_array, all_be_local_vertex,
                                    chunks_out[chunk_index]);
  };

  ThreadGroup tg(comm_spec_);
  for (size_t chunk_index = 0; chunk_index < chunk_num; ++chunk_index) {
    std::shared_ptr<arrow::Array> id_array = id_arrays_in->chunk(chunk_index);
    tg.AddTask(parsefn, chunk_index, id_array);
  }

  Status status;
  for (auto& status : tg.TakeResults()) {
    status += status;
  }
  RETURN_ON_ERROR(status);
  out = std::make_shared<arrow::ChunkedArray>(chunks_out);
  return Status::OK();
}

template <typename OID_T, typename VID_T,
          template <typename, typename> class VERTEX_MAP_T>
vineyard::Status
GARFragmentLoader<OID_T, VID_T, VERTEX_MAP_T>::parseIdChunkedArrayChunk(
    label_id_t label_id, const std::shared_ptr<arrow::Array> id_array_in,
    bool all_be_local_vertex, std::shared_ptr<arrow::Array>& out) {
  std::shared_ptr<arrow::Int64Array> id_array =
      std::dynamic_pointer_cast<arrow::Int64Array>(id_array_in);

  // prepare buffer
  std::unique_ptr<arrow::Buffer> buffer;
  {
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(
        buffer, arrow::AllocateBuffer(id_array->length() * sizeof(vid_t)));
  }

  vid_t* builder = reinterpret_cast<vid_t*>(buffer->mutable_data());
  const auto& label_name = vertex_labels_[label_id];
  const gar_id_t* ids =
      reinterpret_cast<const gar_id_t*>(id_array->raw_values());
  if (all_be_local_vertex) {
    gar_id_t start_id =
        vertex_chunk_begin_of_frag_[label_name][comm_spec_.fid()] *
        vertex_chunk_sizes_[label_id];
    for (int64_t k = 0; k != id_array->length(); ++k) {
      builder[k] =
          vid_parser_.GenerateId(comm_spec_.fid(), label_id, ids[k] - start_id);
    }
  } else {
    for (int64_t k = 0; k != id_array->length(); ++k) {
      fid_t fid = getPartitionId(ids[k], label_id);
      builder[k] = vid_parser_.GenerateId(
          fid, label_id,
          ids[k] - vertex_chunk_begin_of_frag_[label_name][fid] *
                       vertex_chunk_sizes_[label_id]);
    }
  }
  out = std::make_shared<ArrowArrayType<VID_T>>(
      id_array->length(), std::shared_ptr<arrow::Buffer>(std::move(buffer)),
      nullptr, 0);
  return Status::OK();
}

}  // namespace vineyard
#endif  // ENABLE_GAR
#endif  // MODULES_GRAPH_LOADER_GAR_FRAGMENT_LOADER_IMPL_H_
