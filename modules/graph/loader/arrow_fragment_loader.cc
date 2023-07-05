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

#include "graph/loader/arrow_fragment_loader.h"

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
#include "basic/stream/dataframe_stream.h"
#include "basic/stream/parallel_stream.h"
#include "basic/stream/recordbatch_stream.h"
#include "client/client.h"
#include "graph/loader/fragment_loader_utils.h"
#include "graph/utils/error.h"

namespace vineyard {

template <typename LocalStreamT>
static Status ReadRecordBatchesFromVineyardStreamImpl(
    Client& client, Tuple<std::shared_ptr<LocalStreamT>>& local_streams,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, int part_id,
    int part_num) {
  size_t split_size = local_streams.size() / part_num +
                      (local_streams.size() % part_num == 0 ? 0 : 1);
  size_t start_to_read = part_id * split_size;
  size_t end_to_read =
      std::min(local_streams.size(), (part_id + 1) * split_size);

  VLOG(100) << "reading recordbatches from vineyard: total chunks = "
            << local_streams.size() << ", part id = " << part_id
            << ", part num = " << part_num
            << ", start to read = " << start_to_read
            << ", end to read = " << end_to_read
            << ", split size = " << split_size;

  std::mutex mutex_for_results;

  auto reader = [&client, &local_streams, &mutex_for_results,
                 &batches](size_t idx) -> Status {
    // use a local client, since reading from stream may block the client.
    Client local_client;
    RETURN_ON_ERROR(local_client.Connect(client.IPCSocket()));

    auto& stream = local_streams[idx];
    RETURN_ON_ERROR(stream->OpenReader(&local_client));
    std::vector<std::shared_ptr<arrow::RecordBatch>> read_batches;
    RETURN_ON_ERROR(stream->ReadRecordBatches(read_batches));
    {
      std::lock_guard<std::mutex> scoped_lock(mutex_for_results);
      batches.insert(batches.end(), read_batches.begin(), read_batches.end());
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
  size_t total_rows = 0;
  for (auto const& batch : batches) {
    total_rows += batch->num_rows();
  }
  VLOG(10) << "read record batch from vineyard: total rows = " << total_rows;
  return Status::OK();
}

Status ReadRecordBatchesFromVineyardStream(
    Client& client, std::shared_ptr<ParallelStream>& pstream,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, int part_id,
    int part_num) {
  {
    Tuple<std::shared_ptr<RecordBatchStream>> local_streams;
    pstream->GetLocals(local_streams);
    if (!local_streams.empty()) {
      return ReadRecordBatchesFromVineyardStreamImpl<RecordBatchStream>(
          client, local_streams, batches, part_id, part_num);
    }
  }
  {
    Tuple<std::shared_ptr<DataframeStream>> local_streams;
    pstream->GetLocals(local_streams);
    if (!local_streams.empty()) {
      return ReadRecordBatchesFromVineyardStreamImpl<DataframeStream>(
          client, local_streams, batches, part_id, part_num);
    }
  }
  return Status::Invalid("No local partitions in the stream: part_id = " +
                         std::to_string(part_id) +
                         ", part_num = " + std::to_string(part_num) +
                         ", stream = " + pstream->meta().MetaData().dump());
}

Status ReadRecordBatchesFromVineyardDataFrame(
    Client& client, std::shared_ptr<GlobalDataFrame>& gdf,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, int part_id,
    int part_num) {
  std::vector<std::shared_ptr<DataFrame>> local_chunks;
  for (auto iter = gdf->LocalBegin(); iter != gdf->LocalEnd();
       iter.NextLocal()) {
    local_chunks.emplace_back(*iter);
  }
  size_t split_size = local_chunks.size() / part_num +
                      (local_chunks.size() % part_num == 0 ? 0 : 1);
  int start_to_read = part_id * split_size;
  int end_to_read = std::min(local_chunks.size(), (part_id + 1) * split_size);
  for (int idx = start_to_read; idx != end_to_read; ++idx) {
    batches.emplace_back(local_chunks[idx]->AsBatch(true));
  }
  size_t total_rows = 0;
  for (auto const& batch : batches) {
    total_rows += batch->num_rows();
  }
  VLOG(10) << "read record batch from vineyard: total rows = " << total_rows;
  return Status::OK();
}

Status ReadRecordBatchesFromVineyard(
    Client& client, const ObjectID object_id,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, int part_id,
    int part_num) {
  VLOG(10) << "loading table from vineyard: " << ObjectIDToString(object_id)
           << ", part id = " << part_id << ", part num = " << part_num;

  std::shared_ptr<Object> source;
  RETURN_ON_ERROR(client.GetObject(object_id, source));
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

template <typename LocalStreamT>
static Status ReadTableFromVineyardStreamImpl(
    Client& client, Tuple<std::shared_ptr<LocalStreamT>>& local_streams,
    std::shared_ptr<arrow::Table>& table, int part_id, int part_num) {
  size_t split_size = local_streams.size() / part_num +
                      (local_streams.size() % part_num == 0 ? 0 : 1);
  int start_to_read = part_id * split_size;
  int end_to_read = std::min(local_streams.size(), (part_id + 1) * split_size);

  VLOG(10) << "reading table from vineyard: total chunks = "
           << local_streams.size() << ", part id = " << part_id
           << ", part num = " << part_num
           << ", start to read = " << start_to_read
           << ", end to read = " << end_to_read
           << ", split size = " << split_size;

  std::mutex mutex_for_results;
  std::vector<std::shared_ptr<arrow::Table>> tables;
  auto reader = [&client, &local_streams, &mutex_for_results,
                 &tables](size_t idx) -> Status {
    // use a local client, since reading from stream may block the client.
    Client local_client;
    RETURN_ON_ERROR(local_client.Connect(client.IPCSocket()));

    auto const& stream = local_streams[idx];
    RETURN_ON_ERROR(local_streams[idx]->OpenReader(&local_client));
    std::shared_ptr<arrow::Table> table;
    RETURN_ON_ERROR(stream->ReadTable(table));
    if (table == nullptr) {
      VLOG(10) << "table from stream is null.";
    } else {
      VLOG(10) << "table from stream: " << table->schema()->ToString();
      {
        std::lock_guard<std::mutex> scoped_lock(mutex_for_results);
        tables.emplace_back(table);
      }
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
    RETURN_ON_ERROR(ConcatenateTables(tables, table));
  }
  if (table != nullptr) {
    VLOG(10) << "read table from vineyard: total rows = " << table->num_rows();
  } else {
    VLOG(10) << "read table from vineyard: total rows = " << 0;
  }
  return Status::OK();
}

/**
 * @brief When the stream is empty, the result `table` will be set as nullptr.
 */
Status ReadTableFromVineyardStream(Client& client,
                                   std::shared_ptr<ParallelStream>& pstream,
                                   std::shared_ptr<arrow::Table>& table,
                                   int part_id, int part_num) {
  {
    Tuple<std::shared_ptr<RecordBatchStream>> local_streams;
    pstream->GetLocals(local_streams);
    if (!local_streams.empty()) {
      return ReadTableFromVineyardStreamImpl<RecordBatchStream>(
          client, local_streams, table, part_id, part_num);
    }
  }
  {
    Tuple<std::shared_ptr<DataframeStream>> local_streams;
    pstream->GetLocals(local_streams);
    if (!local_streams.empty()) {
      return ReadTableFromVineyardStreamImpl<DataframeStream>(
          client, local_streams, table, part_id, part_num);
    }
  }
  return Status::Invalid("No local partitions in the stream: part_id = " +
                         std::to_string(part_id) +
                         ", part_num = " + std::to_string(part_num) +
                         ", stream = " + pstream->meta().MetaData().dump());
}

/**
 * @brief When no local chunk, the result `table` will be set as nullptr.
 */
Status ReadTableFromVineyardDataFrame(Client& client,
                                      std::shared_ptr<GlobalDataFrame>& gdf,
                                      std::shared_ptr<arrow::Table>& table,
                                      int part_id, int part_num) {
  std::vector<std::shared_ptr<DataFrame>> local_chunks;
  for (auto iter = gdf->LocalBegin(); iter != gdf->LocalEnd();
       iter.NextLocal()) {
    local_chunks.emplace_back(*iter);
  }
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
    VLOG(10) << "read table from vineyard: total rows = " << 0;
    return Status::OK();
  } else {
    auto status = RecordBatchesToTableWithCast(batches, &table);
    if (status.ok()) {
      VLOG(10) << "read table from vineyard: total rows = "
               << table->num_rows();
    } else {
      VLOG(10) << "read table from vineyard: total rows = " << 0;
    }
    return status;
  }
}

/**
 * @brief The result `table` will be set as nullptr.
 */
Status ReadTableFromVineyard(Client& client, const ObjectID object_id,
                             std::shared_ptr<arrow::Table>& table, int part_id,
                             int part_num) {
  VLOG(10) << "loading table from vineyard: " << ObjectIDToString(object_id)
           << ", part id = " << part_id << ", part num = " << part_num;
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

boost::leaf::result<std::vector<std::vector<std::shared_ptr<arrow::Table>>>>
GatherETables(Client& client,
              const std::vector<std::vector<ObjectID>>& estreams, int part_id,
              int part_num) {
  using batch_group_t = std::unordered_map<
      std::string, std::map<std::pair<std::string, std::string>,
                            std::vector<std::shared_ptr<arrow::RecordBatch>>>>;
  batch_group_t grouped_batches;
  std::mutex mutex_for_results;
  auto reader = [&client, &mutex_for_results, &grouped_batches, part_id,
                 part_num](size_t const index,
                           ObjectID const estream) -> Status {
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
        VY_OK_OR_RAISE(RecordBatchesToTableWithCast(subgroup.second, &table));
      }
      subtables.emplace_back(table);
    }
    tables.emplace_back(subtables);
  }
  return tables;
}

boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>> GatherVTables(
    Client& client, const std::vector<ObjectID>& vstreams, int part_id,
    int part_num) {
  using batch_group_t =
      std::unordered_map<std::string,
                         std::vector<std::shared_ptr<arrow::RecordBatch>>>;
  batch_group_t grouped_batches;
  std::mutex mutex_for_results;
  auto reader = [&client, &mutex_for_results, &grouped_batches, part_id,
                 part_num](size_t const index,
                           ObjectID const vstream) -> Status {
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
      VY_OK_OR_RAISE(RecordBatchesToTableWithCast(group.second, &table));
    }
    tables.emplace_back(table);
  }
  return tables;
}

Status ReadTableFromPandas(const std::string& data,
                           std::shared_ptr<arrow::Table>& table) {
  if (!data.empty()) {
    std::shared_ptr<arrow::Buffer> buffer = arrow::Buffer::FromString(data);
    RETURN_ON_ERROR(vineyard::DeserializeTable(buffer, &table));
  }
  return Status::OK();
}

Status ReadTableFromLocation(const std::string& location,
                             std::shared_ptr<arrow::Table>& table, int index,
                             int total_parts) {
  std::string expanded = vineyard::ExpandEnvironmentVariables(location);
  auto io_adaptor = vineyard::IOFactory::CreateIOAdaptor(expanded);
  VINEYARD_ASSERT(io_adaptor != nullptr,
                  "Cannot find a supported adaptor for " + location);
  RETURN_ON_ERROR(io_adaptor->SetPartialRead(index, total_parts));
  RETURN_ON_ERROR(io_adaptor->Open());
  RETURN_ON_ERROR(io_adaptor->ReadTable(&table));

  if (table != nullptr) {  // the file may be too small
    auto meta = std::make_shared<arrow::KeyValueMetadata>();
    for (auto const& item : io_adaptor->GetMeta()) {
      VINEYARD_DISCARD(meta->Set(item.first, item.second));
    }
    auto table_meta = table->schema()->metadata();
    if (table_meta != nullptr) {
      for (auto const& item : table_meta->sorted_pairs()) {
        VINEYARD_DISCARD(meta->Set(item.first, item.second));
      }
    }
    table = table->ReplaceSchemaMetadata(meta);
  }

  RETURN_ON_ERROR(io_adaptor->Close());
  return Status::OK();
}

boost::leaf::result<std::pair<table_vec_t, std::vector<table_vec_t>>>
DataLoader::LoadVertexEdgeTables() {
  BOOST_LEAF_AUTO(v_tables, LoadVertexTables());
  BOOST_LEAF_AUTO(e_tables, LoadEdgeTables());
  return std::make_pair(v_tables, e_tables);
}

boost::leaf::result<table_vec_t> DataLoader::LoadVertexTables() {
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "READ-VERTEX-0";
  table_vec_t v_tables;
  if (!vfiles_.empty()) {
    auto load_v_procedure = [&]() {
      return loadVertexTables(vfiles_, comm_spec_.local_id(),
                              comm_spec_.local_num());
    };
    BOOST_LEAF_ASSIGN(v_tables,
                      vineyard::sync_gs_error(comm_spec_, load_v_procedure));
  } else if (!partial_v_tables_.empty()) {
    v_tables = std::move(partial_v_tables_);
    partial_v_tables_.clear();
  }
  for (const auto& table : v_tables) {
    BOOST_LEAF_CHECK(sanityChecks(table));
  }
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "READ-VERTEX-100";
  return v_tables;
}

boost::leaf::result<std::vector<table_vec_t>> DataLoader::LoadEdgeTables() {
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "READ-EDGE-0";
  std::vector<table_vec_t> e_tables;
  if (!efiles_.empty()) {
    auto load_e_procedure = [&]() {
      return loadEdgeTables(efiles_, comm_spec_.local_id(),
                            comm_spec_.local_num());
    };
    BOOST_LEAF_ASSIGN(e_tables,
                      vineyard::sync_gs_error(comm_spec_, load_e_procedure));
  } else if (!partial_e_tables_.empty()) {
    e_tables = std::move(partial_e_tables_);
    partial_e_tables_.clear();
  }
  for (const auto& table_vec : e_tables) {
    for (const auto& table : table_vec) {
      BOOST_LEAF_CHECK(sanityChecks(table));
    }
  }
  LOG_IF(INFO, !comm_spec_.worker_id()) << MARKER << "READ-EDGE-100";
  return e_tables;
}

boost::leaf::result<ObjectID> DataLoader::resolveVineyardObject(
    std::string const& source) {
  vineyard::ObjectID sourceId = vineyard::InvalidObjectID();
  // encoding: 'o' prefix for object id, and 's' prefix for object name.
  CHECK_OR_RAISE(!source.empty() && (source[0] == 'o' || source[0] == 's'));
  if (source[0] == 'o') {
    sourceId = vineyard::ObjectIDFromString(source.substr(1));
  } else {
    VY_OK_OR_RAISE(client_.GetName(source.substr(1), sourceId, true));
  }
  CHECK_OR_RAISE(sourceId != vineyard::InvalidObjectID());
  return sourceId;
}

boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>>
DataLoader::loadVertexTables(const std::vector<std::string>& files, int index,
                             int total_parts) {
  auto label_num = static_cast<label_id_t>(files.size());
  std::vector<std::shared_ptr<arrow::Table>> tables(label_num);

  auto read_procedure = [&](label_id_t label_id,
                            std::string sub_label_file_name)
      -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
    std::shared_ptr<arrow::Table> table;
    if (files[label_id].rfind("vineyard://", 0) == 0) {
      BOOST_LEAF_AUTO(sourceId,
                      resolveVineyardObject(files[label_id].substr(11)));
      VY_OK_OR_RAISE(
          ReadTableFromVineyard(client_, sourceId, table, index, total_parts));
    } else {
      VY_OK_OR_RAISE(ReadTableFromLocation(sub_label_file_name, table, index,
                                           total_parts));
    }
    return table;
  };

  for (label_id_t label_id = 0; label_id < label_num; ++label_id) {
    std::vector<std::string> sub_label_files;
    boost::split(sub_label_files, files[label_id], boost::is_any_of(";"));
    for (size_t j = 0; j < sub_label_files.size(); ++j) {
      BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, read_procedure, label_id,
                                           sub_label_files[j]));

      if (table == nullptr || table->num_rows() == 0) {
        continue;
      }

      auto sync_schema_procedure =
          [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
        return SyncSchema(table, comm_spec_);
      };

      // normailize the schema of this distributed table
      BOOST_LEAF_AUTO(normalized_table,
                      sync_gs_error(comm_spec_, sync_schema_procedure));

      auto meta = normalized_table->schema()->metadata();
      if (meta == nullptr || meta->FindKey(LABEL_TAG) == -1) {
        RETURN_GS_ERROR(
            ErrorCode::kIOError,
            "Metadata of input vertex files should contain label name");
      }

      if (j == 0) {
        tables[label_id] = normalized_table;
        continue;
      }
      std::shared_ptr<arrow::Table> prev_table = tables[label_id];
      std::shared_ptr<arrow::Table> combined_table;
      ConcatenateTables({prev_table, normalized_table}, combined_table);
      tables[label_id] = combined_table;
    }
  }
  return tables;
}

boost::leaf::result<std::vector<std::vector<std::shared_ptr<arrow::Table>>>>
DataLoader::loadEdgeTables(const std::vector<std::string>& files, int index,
                           int total_parts) {
  auto label_num = static_cast<label_id_t>(files.size());
  std::vector<std::vector<std::shared_ptr<arrow::Table>>> tables(label_num);

  try {
    auto read_procedure = [&](label_id_t label_id,
                              std::string sub_label_file_name)
        -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
      std::shared_ptr<arrow::Table> table;
      if (files[label_id].rfind("vineyard://", 0) == 0) {
        BOOST_LEAF_AUTO(sourceId,
                        resolveVineyardObject(files[label_id].substr(11)));
        VY_OK_OR_RAISE(ReadTableFromVineyard(client_, sourceId, table, index,
                                             total_parts));
      } else {
        VY_OK_OR_RAISE(ReadTableFromLocation(sub_label_file_name, table, index,
                                             total_parts));
      }
      return table;
    };

    for (label_id_t label_id = 0; label_id < label_num; ++label_id) {
      std::vector<std::string> sub_label_files;
      boost::split(sub_label_files, files[label_id], boost::is_any_of(";"));
      for (size_t j = 0; j < sub_label_files.size(); ++j) {
        BOOST_LEAF_AUTO(table, sync_gs_error(comm_spec_, read_procedure,
                                             label_id, sub_label_files[j]));
        if (table == nullptr || table->num_rows() == 0) {
          continue;
        }
        // normailize the schema of this distributed table
        auto sync_schema_procedure =
            [&]() -> boost::leaf::result<std::shared_ptr<arrow::Table>> {
          return SyncSchema(table, comm_spec_);
        };
        BOOST_LEAF_AUTO(normalized_table,
                        sync_gs_error(comm_spec_, sync_schema_procedure));

        auto meta = normalized_table->schema()->metadata();
        if (meta == nullptr || meta->FindKey(LABEL_TAG) == -1) {
          RETURN_GS_ERROR(
              ErrorCode::kIOError,
              "Metadata of input edge files should contain label name");
        }
        if (meta == nullptr || meta->FindKey(SRC_LABEL_TAG) == -1) {
          RETURN_GS_ERROR(
              ErrorCode::kIOError,
              "Metadata of input edge files should contain src label name");
        }
        if (meta == nullptr || meta->FindKey(DST_LABEL_TAG) == -1) {
          RETURN_GS_ERROR(
              ErrorCode::kIOError,
              "Metadata of input edge files should contain dst label name");
        }

        tables[label_id].emplace_back(normalized_table);
      }
    }
  } catch (std::exception& e) {
    RETURN_GS_ERROR(ErrorCode::kIOError, std::string(e.what()));
  }
  return tables;
}

boost::leaf::result<void> DataLoader::sanityChecks(
    std::shared_ptr<arrow::Table> table) {
  // We require that there are no identical column names
  auto names = table->ColumnNames();
  std::sort(names.begin(), names.end());
  const auto duplicate = std::adjacent_find(names.begin(), names.end());
  if (duplicate != names.end()) {
    auto meta = table->schema()->metadata();
    int label_meta_index = meta->FindKey(LABEL_TAG);
    std::string label_name = meta->value(label_meta_index);
    std::stringstream msg;
    msg << "Label " << label_name
        << " has identical property names, which is not allowed. The "
           "original names are: ";
    auto origin_names = table->ColumnNames();
    msg << "[";
    for (size_t i = 0; i < origin_names.size(); ++i) {
      if (i != 0) {
        msg << ", ";
      }
      msg << origin_names[i];
    }
    msg << "]";
    RETURN_GS_ERROR(vineyard::ErrorCode::kInvalidValueError, msg.str());
  }
  return {};
}

}  // namespace vineyard
