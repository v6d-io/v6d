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

#ifndef MODULES_GRAPH_LOADER_ARROW_FRAGMENT_LOADER_H_
#define MODULES_GRAPH_LOADER_ARROW_FRAGMENT_LOADER_H_

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "grape/worker/comm_spec.h"

#include "basic/ds/dataframe.h"
#include "basic/ds/tensor.h"
#include "basic/stream/parallel_stream.h"
#include "basic/stream/recordbatch_stream.h"
#include "client/client.h"
#include "io/io/io_factory.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/graph_schema.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/fragment/property_graph_utils.h"
#include "graph/loader/basic_ev_fragment_loader.h"
#include "graph/loader/fragment_loader_utils.h"
#include "graph/utils/error.h"
#include "graph/utils/partitioner.h"
#include "graph/utils/thread_group.h"
#include "graph/vertex_map/arrow_vertex_map.h"

#define HASH_PARTITION

namespace vineyard {

inline Status ReadRecordBatchesFromVineyardStream(
    Client& client, std::shared_ptr<ParallelStream>& pstream,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, int part_id,
    int part_num) {
  Tuple<std::shared_ptr<RecordBatchStream>> local_streams;
  pstream->GetLocals(local_streams);

  size_t split_size = local_streams.size() / part_num +
                      (local_streams.size() % part_num == 0 ? 0 : 1);
  size_t start_to_read = part_id * split_size;
  size_t end_to_read =
      std::min(local_streams.size(), (part_id + 1) * split_size);

  std::mutex mutex_for_results;

  auto reader = [&client, &local_streams, &mutex_for_results,
                 &batches](size_t idx) {
    // use a local client, since reading from stream may block the client.
    Client local_client;
    RETURN_ON_ERROR(local_client.Connect(client.IPCSocket()));

    auto& stream = local_streams[idx];
    VINEYARD_CHECK_OK(stream->OpenReader(&local_client));
    std::vector<std::shared_ptr<arrow::RecordBatch>> read_batches;
    RETURN_ON_ERROR(stream->ReadRecordBatches(read_batches));
    {
      std::lock_guard<std::mutex> scoped_lock(mutex_for_results);
      for (auto const& batch : read_batches) {
        batches.emplace_back(batch);
      }
    }
    return Status::OK();
  };

  ThreadGroup tg;
  for (size_t idx = start_to_read; idx != end_to_read; ++idx) {
    tg.AddTask(reader, idx);
  }
  auto readers_status = tg.TakeResults();
  for (auto const& status : readers_status) {
    RETURN_ON_ERROR(status);
  }
  return Status::OK();
}

inline Status ReadRecordBatchesFromVineyardDataFrame(
    Client& client, std::shared_ptr<GlobalDataFrame>& gdf,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, int part_id,
    int part_num) {
  auto local_chunks = gdf->LocalPartitions(client);
  size_t split_size = local_chunks.size() / part_num +
                      (local_chunks.size() % part_num == 0 ? 0 : 1);
  int start_to_read = part_id * split_size;
  int end_to_read = std::min(local_chunks.size(), (part_id + 1) * split_size);
  for (int idx = start_to_read; idx != end_to_read; ++idx) {
    batches.emplace_back(local_chunks[idx]->AsBatch(true));
  }
  return Status::OK();
}

inline Status ReadRecordBatchesFromVineyard(
    Client& client, const ObjectID object_id,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, int part_id,
    int part_num) {
  auto source = client.GetObject(object_id);
  RETURN_ON_ASSERT(source != nullptr,
                   "Object not exists: " + ObjectIDToString(object_id));
  if (auto pstream = std::dynamic_pointer_cast<ParallelStream>(source)) {
    return ReadRecordBatchesFromVineyardStream(client, pstream, batches,
                                               part_id, part_num);
  }
  if (auto gdf = std::dynamic_pointer_cast<GlobalDataFrame>(source)) {
    return ReadRecordBatchesFromVineyardDataFrame(client, gdf, batches, part_id,
                                                  part_num);
  }

  return Status::Invalid(
      "The source is not a parallel stream nor a global dataframe: " +
      source->meta().GetTypeName());
}

/**
 * @brief When the stream is empty, the result `table` will be set as nullptr.
 */
inline Status ReadTableFromVineyardStream(
    Client& client, std::shared_ptr<ParallelStream>& pstream,
    std::shared_ptr<arrow::Table>& table, int part_id, int part_num) {
  Tuple<std::shared_ptr<RecordBatchStream>> local_streams;
  pstream->GetLocals(local_streams);
  size_t split_size = local_streams.size() / part_num +
                      (local_streams.size() % part_num == 0 ? 0 : 1);
  int start_to_read = part_id * split_size;
  int end_to_read = std::min(local_streams.size(), (part_id + 1) * split_size);
  std::mutex mutex_for_results;
  std::vector<std::shared_ptr<arrow::Table>> tables;
  auto reader = [&client, &local_streams, &mutex_for_results,
                 &tables](size_t idx) {
    // use a local client, since reading from stream may block the client.
    Client local_client;
    RETURN_ON_ERROR(local_client.Connect(client.IPCSocket()));

    auto const& stream = local_streams[idx];
    VINEYARD_CHECK_OK(local_streams[idx]->OpenReader(&local_client));
    std::shared_ptr<arrow::Table> table;
    RETURN_ON_ERROR(stream->ReadTable(table));
    if (table == nullptr) {
      VLOG(10) << "table from stream is null.";
    } else {
      VLOG(10) << "table from stream: " << table->schema()->ToString();
    }
    {
      std::lock_guard<std::mutex> scoped_lock(mutex_for_results);
      tables.emplace_back(table);
    }
    return Status::OK();
  };
  ThreadGroup tg;
  for (int idx = start_to_read; idx != end_to_read; ++idx) {
    tg.AddTask(reader, idx);
  }
  auto readers_status = tg.TakeResults();
  for (auto const& status : readers_status) {
    RETURN_ON_ERROR(status);
  }
  if (tables.empty()) {
    table = nullptr;
  } else {
    table = ConcatenateTables(tables);
  }
  return Status::OK();
}

/**
 * @brief When no local chunk, the result `table` will be set as nullptr.
 */
inline Status ReadTableFromVineyardDataFrame(
    Client& client, std::shared_ptr<GlobalDataFrame>& gdf,
    std::shared_ptr<arrow::Table>& table, int part_id, int part_num) {
  auto local_chunks = gdf->LocalPartitions(client);
  size_t split_size = local_chunks.size() / part_num +
                      (local_chunks.size() % part_num == 0 ? 0 : 1);
  int start_to_read = part_id * split_size;
  int end_to_read = std::min(local_chunks.size(), (part_id + 1) * split_size);
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  batches.reserve(end_to_read - start_to_read);
  for (int idx = start_to_read; idx != end_to_read; ++idx) {
    batches.emplace_back(local_chunks[idx]->AsBatch(true));
  }
  if (batches.empty()) {
    table = nullptr;
    return Status::OK();
  } else {
    return RecordBatchesToTable(batches, &table);
  }
}

/**
 * @brief The result `table` will be set as nullptr.
 */
inline Status ReadTableFromVineyard(Client& client, const ObjectID object_id,
                                    std::shared_ptr<arrow::Table>& table,
                                    int part_id, int part_num) {
  auto source = client.GetObject(object_id);
  RETURN_ON_ASSERT(source != nullptr,
                   "Object not exists: " + ObjectIDToString(object_id));
  if (auto pstream = std::dynamic_pointer_cast<ParallelStream>(source)) {
    return ReadTableFromVineyardStream(client, pstream, table, part_id,
                                       part_num);
  }
  if (auto gdf = std::dynamic_pointer_cast<GlobalDataFrame>(source)) {
    return ReadTableFromVineyardDataFrame(client, gdf, table, part_id,
                                          part_num);
  }

  return Status::Invalid(
      "The source is not a parallel stream nor a global dataframe: " +
      source->meta().GetTypeName());
}

/** Note [GatherETables and GatherVTables]
 *
 * GatherETables and GatherVTables gathers all edges and vertices as table from
 * multiple streams.
 *
 * It requires (one of the follows):
 *
 * + all chunks in the stream has a "label" (and "src_label", "dst_label" for
 *   edges) in meta, and at least one batch available on each worker.
 *
 * + or all chunks doesn't have such meta.
 */

inline boost::leaf::result<
    std::vector<std::vector<std::shared_ptr<arrow::Table>>>>
GatherETables(Client& client,
              const std::vector<std::vector<ObjectID>>& estreams, int part_id,
              int part_num) {
  using batch_group_t = std::unordered_map<
      std::string, std::map<std::pair<std::string, std::string>,
                            std::vector<std::shared_ptr<arrow::RecordBatch>>>>;
  batch_group_t grouped_batches;
  std::mutex mutex_for_results;
  auto reader = [&client, &mutex_for_results, &grouped_batches, part_id,
                 part_num](size_t const index, ObjectID const estream) {
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    auto status = ReadRecordBatchesFromVineyard(client, estream, batches,
                                                part_id, part_num);
    if (status.ok()) {
      std::lock_guard<std::mutex> scoped_lock(mutex_for_results);
      std::string label = std::to_string(index), src_label = "", dst_label = "";
      for (auto const& batch : batches) {
        auto metadata = batch->schema()->metadata();
        if (metadata != nullptr) {
          std::unordered_map<std::string, std::string> meta_map;
          metadata->ToUnorderedMap(&meta_map);
          if (meta_map.find("label") != meta_map.end()) {
            label = meta_map["label"];
          }
          src_label = meta_map["src_label"];
          dst_label = meta_map["dst_label"];
        }
        grouped_batches[label][std::make_pair(src_label, dst_label)]
            .emplace_back(batch);
      }
    } else {
      LOG(ERROR) << "Failed to read from stream " << ObjectIDToString(estream)
                 << ": " << status.ToString();
    }
    return Status::OK();
  };

  ThreadGroup tg;
  for (size_t index = 0; index < estreams.size(); ++index) {
    for (auto const& estream : estreams[index]) {
      tg.AddTask(reader, index, estream);
    }
  }
  tg.TakeResults();

  if (!estreams.empty() && grouped_batches.empty()) {
    grouped_batches[std::to_string(0)][std::make_pair("", "")] = {};
  }

  std::vector<std::vector<std::shared_ptr<arrow::Table>>> tables;
  for (auto const& group : grouped_batches) {
    std::shared_ptr<arrow::Table> table;
    std::vector<std::shared_ptr<arrow::Table>> subtables;
    for (auto const& subgroup : group.second) {
      if (subgroup.second.empty()) {
        table = nullptr;  // no tables at current worker
      } else {
        VY_OK_OR_RAISE(RecordBatchesToTable(subgroup.second, &table));
      }
      subtables.emplace_back(table);
    }
    tables.emplace_back(subtables);
  }
  return tables;
}

inline boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>>
GatherVTables(Client& client, const std::vector<ObjectID>& vstreams,
              int part_id, int part_num) {
  using batch_group_t =
      std::unordered_map<std::string,
                         std::vector<std::shared_ptr<arrow::RecordBatch>>>;
  batch_group_t grouped_batches;
  std::mutex mutex_for_results;
  auto reader = [&client, &mutex_for_results, &grouped_batches, part_id,
                 part_num](size_t const index, ObjectID const vstream) {
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    auto status = ReadRecordBatchesFromVineyard(client, vstream, batches,
                                                part_id, part_num);
    if (status.ok()) {
      std::lock_guard<std::mutex> scoped_lock(mutex_for_results);
      for (auto const& batch : batches) {
        std::string label = std::to_string(index);
        if (batch->schema()->metadata() != nullptr) {
          std::unordered_map<std::string, std::string> meta_map;
          batch->schema()->metadata()->ToUnorderedMap(&meta_map);
          if (meta_map.find("label") != meta_map.end()) {
            label = meta_map["label"];
          }
        }
        grouped_batches[label].emplace_back(batch);
      }
    } else {
      LOG(ERROR) << "Failed to read from stream " << ObjectIDToString(vstream)
                 << ": " << status.ToString();
    }
    return Status::OK();
  };

  ThreadGroup tg;
  for (size_t index = 0; index < vstreams.size(); ++index) {
    tg.AddTask(reader, index, vstreams[index]);
  }
  tg.TakeResults();

  if (!vstreams.empty() && grouped_batches.empty()) {
    grouped_batches[std::to_string(0)] = {};
  }

  std::vector<std::shared_ptr<arrow::Table>> tables;
  for (auto const& group : grouped_batches) {
    std::shared_ptr<arrow::Table> table;
    if (group.second.empty()) {
      table = nullptr;  // no tables at current worker
    } else {
      VY_OK_OR_RAISE(RecordBatchesToTable(group.second, &table));
    }
    tables.emplace_back(table);
  }
  return tables;
}

template <typename OID_T = property_graph_types::OID_TYPE,
          typename VID_T = property_graph_types::VID_TYPE>
class ArrowFragmentLoader {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_t = typename ConvertToArrowType<oid_t>::ArrayType;
  using vertex_map_t = ArrowVertexMap<internal_oid_t, vid_t>;
  // These consts represent the key in the path of vfile/efile
  static constexpr const char* LABEL_TAG = "label";
  static constexpr const char* SRC_LABEL_TAG = "src_label";
  static constexpr const char* DST_LABEL_TAG = "dst_label";

  static constexpr int id_column = 0;
#ifdef HASH_PARTITION
  using partitioner_t = HashPartitioner<oid_t>;
#else
  using partitioner_t = SegmentedPartitioner<oid_t>;
#endif
  using vertex_table_info_t =
      std::map<std::string, std::shared_ptr<arrow::Table>>;
  using edge_table_info_t = std::vector<InputTable>;

 public:
  /**
   *
   * @param client
   * @param comm_spec
   * @param efiles An example of efile:
   * /data/twitter_e_0_0_0#src_label=v0&dst_label=v0&label=e0;/data/twitter_e_0_1_0#src_label=v0&dst_label=v1&label=e0;/data/twitter_e_1_0_0#src_label=v1&dst_label=v0&label=e0;/data/twitter_e_1_1_0#src_label=v1&dst_label=v1&label=e0
   * @param vfiles An example of vfile: /data/twitter_v_0#label=v0
   * @param directed
   */
  ArrowFragmentLoader(Client& client, const grape::CommSpec& comm_spec,
                      const std::vector<std::string>& efiles,
                      const std::vector<std::string>& vfiles,
                      bool directed = true, bool generate_eid = false)
      : client_(client),
        comm_spec_(comm_spec),
        efiles_(efiles),
        vfiles_(vfiles),
        directed_(directed),
        generate_eid_(generate_eid) {}

  ArrowFragmentLoader(Client& client, const grape::CommSpec& comm_spec,
                      const std::vector<std::string>& efiles,
                      bool directed = true, bool generate_eid = false)
      : client_(client),
        comm_spec_(comm_spec),
        efiles_(efiles),
        vfiles_(),
        directed_(directed),
        generate_eid_(generate_eid) {}

  ArrowFragmentLoader(Client& client, const grape::CommSpec& comm_spec,
                      const std::vector<ObjectID>& vstreams,
                      const std::vector<std::vector<ObjectID>>& estreams,
                      bool directed = true, bool generate_eid = false)
      : client_(client),
        comm_spec_(comm_spec),
        v_streams_(vstreams),
        e_streams_(estreams),
        directed_(directed),
        generate_eid_(generate_eid) {}

  ArrowFragmentLoader(
      Client& client, const grape::CommSpec& comm_spec,
      std::vector<std::shared_ptr<arrow::Table>> const& partial_v_tables,
      std::vector<std::vector<std::shared_ptr<arrow::Table>>> const&
          partial_e_tables,
      bool directed = true, bool generate_eid = false)
      : client_(client),
        comm_spec_(comm_spec),
        partial_v_tables_(partial_v_tables),
        partial_e_tables_(partial_e_tables),
        directed_(directed),
        generate_eid_(generate_eid) {}

  ArrowFragmentLoader(
      Client& client, const grape::CommSpec& comm_spec,
      std::vector<std::vector<std::shared_ptr<arrow::Table>>> const&
          partial_e_tables,
      bool directed = true, bool generate_eid = false)
      : client_(client),
        comm_spec_(comm_spec),
        partial_v_tables_(),
        partial_e_tables_(partial_e_tables),
        directed_(directed),
        generate_eid_(generate_eid) {}

  ~ArrowFragmentLoader() = default;

  boost::leaf::result<ObjectID> LoadFragment() {
    BOOST_LEAF_CHECK(initPartitioner());

    std::vector<std::shared_ptr<arrow::Table>> partial_v_tables;
    std::vector<std::vector<std::shared_ptr<arrow::Table>>> partial_e_tables;
    if (!v_streams_.empty() && !e_streams_.empty()) {
      {
        BOOST_LEAF_AUTO(
            tmp, GatherVTables(client_, v_streams_, comm_spec_.local_id(),
                               comm_spec_.local_num()));
        partial_v_tables = tmp;
      }
      {
        BOOST_LEAF_AUTO(
            tmp, GatherETables(client_, e_streams_, comm_spec_.local_id(),
                               comm_spec_.local_num()));
        partial_e_tables = tmp;
      }
    } else if (!vfiles_.empty() && !efiles_.empty()) {
      auto load_v_procedure = [&]() {
        return loadVertexTables(vfiles_, comm_spec_.worker_id(),
                                comm_spec_.worker_num());
      };
      BOOST_LEAF_AUTO(tmp_v, sync_gs_error(comm_spec_, load_v_procedure));
      partial_v_tables = tmp_v;
      auto load_e_procedure = [&]() {
        return loadEdgeTables(efiles_, comm_spec_.worker_id(),
                              comm_spec_.worker_num());
      };
      BOOST_LEAF_AUTO(tmp_e, sync_gs_error(comm_spec_, load_e_procedure));
      partial_e_tables = tmp_e;
    } else if (vfiles_.empty() && !efiles_.empty()) {
      auto load_e_procedure = [&]() {
        return loadEdgeTables(efiles_, comm_spec_.worker_id(),
                              comm_spec_.worker_num());
      };
      BOOST_LEAF_AUTO(tmp_e, sync_gs_error(comm_spec_, load_e_procedure));
      partial_e_tables = tmp_e;
    } else if (!partial_e_tables_.empty() && !partial_e_tables_.empty()) {
      for (size_t vlabel = 0; vlabel < partial_v_tables_.size(); ++vlabel) {
        std::shared_ptr<arrow::Table> result_table;
        partial_v_tables.emplace_back(partial_v_tables_[vlabel]);
      }
      for (size_t elabel = 0; elabel < partial_e_tables_.size(); ++elabel) {
        std::vector<std::shared_ptr<arrow::Table>> subetables;
        for (auto const& etable : partial_e_tables_[elabel]) {
          subetables.emplace_back(etable);
        }
        partial_e_tables.emplace_back(subetables);
      }
    } else if (partial_v_tables_.empty() && !partial_e_tables_.empty()) {
      for (size_t elabel = 0; elabel < partial_e_tables_.size(); ++elabel) {
        std::vector<std::shared_ptr<arrow::Table>> subetables;
        for (auto const& etable : partial_e_tables_[elabel]) {
          subetables.emplace_back(etable);
        }
        partial_e_tables.emplace_back(subetables);
      }
    } else {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Error when processing input source");
    }

    std::shared_ptr<BasicEVFragmentLoader<OID_T, VID_T, partitioner_t>>
        basic_fragment_loader = std::make_shared<
            BasicEVFragmentLoader<OID_T, VID_T, partitioner_t>>(
            client_, comm_spec_, partitioner_, directed_, true, generate_eid_);

    BOOST_LEAF_AUTO(v_e_tables,
                    preprocessInputs(partial_v_tables, partial_e_tables));

    auto vertex_tables_with_label = v_e_tables.first;
    auto edge_tables_with_label = v_e_tables.second;

    for (auto& pair : vertex_tables_with_label) {
      BOOST_LEAF_CHECK(
          basic_fragment_loader->AddVertexTable(pair.first, pair.second));
    }
    BOOST_LEAF_CHECK(basic_fragment_loader->ConstructVertices());

    partial_v_tables.clear();
    vertex_tables_with_label.clear();

    for (auto& table : edge_tables_with_label) {
      BOOST_LEAF_CHECK(basic_fragment_loader->AddEdgeTable(
          table.src_label, table.dst_label, table.edge_label, table.table));
    }
    partial_e_tables_.clear();
    edge_tables_with_label.clear();

    BOOST_LEAF_CHECK(basic_fragment_loader->ConstructEdges());

    return basic_fragment_loader->ConstructFragment();
  }

  boost::leaf::result<ObjectID> LoadFragmentAsFragmentGroup() {
    BOOST_LEAF_AUTO(frag_id, LoadFragment());
    auto frag = std::dynamic_pointer_cast<ArrowFragment<OID_T, VID_T>>(
        client_.GetObject(frag_id));

    BOOST_LEAF_AUTO(group_id,
                    ConstructFragmentGroup(client_, frag_id, comm_spec_));
    return group_id;
  }

 protected:  // for subclasses
  boost::leaf::result<void> initPartitioner() {
#ifdef HASH_PARTITION
    partitioner_.Init(comm_spec_.fnum());
#else
    if (vfiles_.empty()) {
      RETURN_GS_ERROR(
          ErrorCode::kInvalidOperationError,
          "Segmented partitioner is not supported when the v-file is "
          "not provided");
    }
    std::vector<std::shared_ptr<arrow::Table>> vtables;
    {
      BOOST_LEAF_AUTO(tmp, loadVertexTables(vfiles_, 0, 1));
      vtables = tmp;
    }
    std::vector<oid_t> oid_list;

    for (auto& table : vtables) {
      std::shared_ptr<arrow::ChunkedArray> oid_array_chunks =
          table->column(id_column);
      size_t chunk_num = oid_array_chunks->num_chunks();

      for (size_t chunk_i = 0; chunk_i != chunk_num; ++chunk_i) {
        std::shared_ptr<oid_array_t> array =
            std::dynamic_pointer_cast<oid_array_t>(
                oid_array_chunks->chunk(chunk_i));
        int64_t length = array->length();
        for (int64_t i = 0; i < length; ++i) {
          oid_list.emplace_back(oid_t(array->GetView(i)));
        }
      }
    }

    partitioner_.Init(comm_spec_.fnum(), oid_list);
#endif
    return {};
  }

  boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>>
  loadVertexTables(const std::vector<std::string>& files, int index,
                   int total_parts) {
    auto label_num = static_cast<label_id_t>(files.size());
    std::vector<std::shared_ptr<arrow::Table>> tables(label_num);

    for (label_id_t label_id = 0; label_id < label_num; ++label_id) {
      std::unique_ptr<IIOAdaptor, std::function<void(IIOAdaptor*)>> io_adaptor(
          IOFactory::CreateIOAdaptor(files[label_id] + "#header_row=true")
              .release(),
          io_deleter_);
      auto read_procedure =
          [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
        VY_OK_OR_RAISE(io_adaptor->SetPartialRead(index, total_parts));
        VY_OK_OR_RAISE(io_adaptor->Open());
        std::shared_ptr<arrow::Table> table;
        VY_OK_OR_RAISE(io_adaptor->ReadTable(&table));
        return table;
      };

      BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, read_procedure));

      auto sync_schema_procedure =
          [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
        return SyncSchema(table, comm_spec_);
      };

      BOOST_LEAF_AUTO(normalized_table,
                      sync_gs_error(comm_spec_, sync_schema_procedure));

      auto meta = std::make_shared<arrow::KeyValueMetadata>();

      auto adaptor_meta = io_adaptor->GetMeta();
      // Check if label name is in meta
      if (adaptor_meta.find(LABEL_TAG) == adaptor_meta.end()) {
        RETURN_GS_ERROR(
            ErrorCode::kIOError,
            "Metadata of input vertex files should contain label name");
      }
      auto v_label_name = adaptor_meta.find(LABEL_TAG)->second;

      CHECK_ARROW_ERROR(meta->Set(LABEL_TAG, v_label_name));

      tables[label_id] = normalized_table->ReplaceSchemaMetadata(meta);
    }
    return tables;
  }

  boost::leaf::result<std::vector<std::vector<std::shared_ptr<arrow::Table>>>>
  loadEdgeTables(const std::vector<std::string>& files, int index,
                 int total_parts) {
    auto label_num = static_cast<label_id_t>(files.size());
    std::vector<std::vector<std::shared_ptr<arrow::Table>>> tables(label_num);

    try {
      for (label_id_t label_id = 0; label_id < label_num; ++label_id) {
        std::vector<std::string> sub_label_files;
        boost::split(sub_label_files, files[label_id], boost::is_any_of(";"));

        for (size_t j = 0; j < sub_label_files.size(); ++j) {
          std::unique_ptr<IIOAdaptor, std::function<void(IIOAdaptor*)>>
              io_adaptor(IOFactory::CreateIOAdaptor(sub_label_files[j] +
                                                    "#header_row=true")
                             .release(),
                         io_deleter_);
          auto read_procedure =
              [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
            VY_OK_OR_RAISE(io_adaptor->SetPartialRead(index, total_parts));
            VY_OK_OR_RAISE(io_adaptor->Open());
            std::shared_ptr<arrow::Table> table;
            VY_OK_OR_RAISE(io_adaptor->ReadTable(&table));
            return table;
          };
          BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, read_procedure));

          auto sync_schema_procedure =
              [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
            return SyncSchema(table, comm_spec_);
          };
          BOOST_LEAF_AUTO(normalized_table,
                          sync_gs_error(comm_spec_, sync_schema_procedure));

          std::shared_ptr<arrow::KeyValueMetadata> meta(
              new arrow::KeyValueMetadata());

          auto adaptor_meta = io_adaptor->GetMeta();
          auto it = adaptor_meta.find(LABEL_TAG);
          if (it == adaptor_meta.end()) {
            RETURN_GS_ERROR(
                ErrorCode::kIOError,
                "Metadata of input edge files should contain label name");
          }
          std::string edge_label_name = it->second;

          it = adaptor_meta.find(SRC_LABEL_TAG);
          if (it == adaptor_meta.end()) {
            RETURN_GS_ERROR(
                ErrorCode::kIOError,
                "Metadata of input edge files should contain src label name");
          }
          std::string src_label_name = it->second;

          it = adaptor_meta.find(DST_LABEL_TAG);
          if (it == adaptor_meta.end()) {
            RETURN_GS_ERROR(
                ErrorCode::kIOError,
                "Metadata of input edge files should contain dst label name");
          }
          std::string dst_label_name = it->second;

          CHECK_ARROW_ERROR(meta->Set(LABEL_TAG, edge_label_name));
          CHECK_ARROW_ERROR(meta->Set(SRC_LABEL_TAG, src_label_name));
          CHECK_ARROW_ERROR(meta->Set(DST_LABEL_TAG, dst_label_name));

          tables[label_id].emplace_back(
              normalized_table->ReplaceSchemaMetadata(meta));
        }
      }
    } catch (std::exception& e) {
      RETURN_GS_ERROR(ErrorCode::kIOError, std::string(e.what()));
    }
    return tables;
  }

  boost::leaf::result<std::pair<vertex_table_info_t, edge_table_info_t>>
  preprocessInputs(
      const std::vector<std::shared_ptr<arrow::Table>>& v_tables,
      const std::vector<std::vector<std::shared_ptr<arrow::Table>>>& e_tables,
      const std::set<std::string>& previous_vertex_labels =
          std::set<std::string>()) {
    vertex_table_info_t vertex_tables_with_label;
    edge_table_info_t edge_tables_with_label;
    std::set<std::string> deduced_labels;
    for (auto table : v_tables) {
      auto meta = table->schema()->metadata();
      if (meta == nullptr) {
        RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                        "Metadata of input vertex files shouldn't be empty");
      }

      int label_meta_index = meta->FindKey(LABEL_TAG);
      if (label_meta_index == -1) {
        RETURN_GS_ERROR(
            ErrorCode::kInvalidValueError,
            "Metadata of input vertex files should contain label name");
      }
      std::string label_name = meta->value(label_meta_index);
      vertex_tables_with_label[label_name] = table;
    }

    auto label_not_exists = [&](const std::string& label) {
      return vertex_tables_with_label.find(label) ==
                 vertex_tables_with_label.end() &&
             previous_vertex_labels.find(label) == previous_vertex_labels.end();
    };

    for (auto& table_vec : e_tables) {
      for (auto table : table_vec) {
        auto meta = table->schema()->metadata();
        int label_meta_index = meta->FindKey(LABEL_TAG);
        std::string label_name = meta->value(label_meta_index);
        int src_label_meta_index = meta->FindKey(SRC_LABEL_TAG);
        std::string src_label_name = meta->value(src_label_meta_index);
        int dst_label_meta_index = meta->FindKey(DST_LABEL_TAG);
        std::string dst_label_name = meta->value(dst_label_meta_index);
        edge_tables_with_label.emplace_back(src_label_name, dst_label_name,
                                            label_name, table);
        // Find vertex labels that need to be deduced, i.e. not assigned by user
        // directly

        if (label_not_exists(src_label_name)) {
          deduced_labels.insert(src_label_name);
        }
        if (label_not_exists(dst_label_name)) {
          deduced_labels.insert(dst_label_name);
        }
      }
    }

    if (!deduced_labels.empty()) {
      FragmentLoaderUtils<OID_T, VID_T, partitioner_t> loader_utils(
          comm_spec_, partitioner_);
      BOOST_LEAF_AUTO(vertex_labels,
                      loader_utils.GatherVertexLabels(edge_tables_with_label));
      BOOST_LEAF_AUTO(vertex_label_to_index,
                      loader_utils.GetVertexLabelToIndex(vertex_labels));
      BOOST_LEAF_AUTO(v_tables_map, loader_utils.BuildVertexTableFromEdges(
                                        edge_tables_with_label,
                                        vertex_label_to_index, deduced_labels));
      for (auto& pair : v_tables_map) {
        vertex_tables_with_label[pair.first] = pair.second;
      }
    }
    return std::make_pair(vertex_tables_with_label, edge_tables_with_label);
  }

  Client& client_;
  grape::CommSpec comm_spec_;
  std::vector<std::string> efiles_, vfiles_;

  std::vector<ObjectID> v_streams_;
  std::vector<std::vector<ObjectID>> e_streams_;
  std::vector<std::shared_ptr<arrow::Table>> partial_v_tables_;
  std::vector<std::vector<std::shared_ptr<arrow::Table>>> partial_e_tables_;

  partitioner_t partitioner_;

  bool directed_;
  bool generate_eid_;

  std::function<void(IIOAdaptor*)> io_deleter_ = [](IIOAdaptor* adaptor) {
    VINEYARD_CHECK_OK(adaptor->Close());
    delete adaptor;
  };
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_LOADER_ARROW_FRAGMENT_LOADER_H_
