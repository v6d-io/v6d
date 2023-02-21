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
    auto status = RecordBatchesToTable(batches, &table);
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
        VY_OK_OR_RAISE(RecordBatchesToTable(subgroup.second, &table));
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
      VY_OK_OR_RAISE(RecordBatchesToTable(group.second, &table));
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
    auto adaptor_meta = io_adaptor->GetMeta();
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

}  // namespace vineyard
