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

#ifndef MODULES_GRAPH_UTILS_TABLE_PIPELINE_H_
#define MODULES_GRAPH_UTILS_TABLE_PIPELINE_H_

#include <future>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "grape/utils/concurrent_queue.h"

#include "basic/ds/arrow_utils.h"
#include "common/util/status.h"
#include "graph/utils/thread_group.h"

namespace vineyard {

/**
 * @brief A table pipeline likes table, but can only be traversed in pull-based
 *        execution.
 *
 */
class ITablePipeline {
 protected:
  ITablePipeline() = default;
  virtual ~ITablePipeline() = default;

 public:
  virtual Status Next(std::shared_ptr<arrow::RecordBatch>& batch) = 0;

  std::shared_ptr<arrow::Schema> schema() const { return schema_; }

  int64_t length() const { return length_; }

  int64_t num_columns() const { return schema_->num_fields(); }

  int64_t num_batches() const { return num_batches_; }

  // TODO
  // void ReplaceSchemaMetadata(const
  // std::shared_ptr<arrow::KeyValueMetadata>())

 protected:
  std::shared_ptr<arrow::Schema> schema_;
  int64_t length_ = -1;
  int64_t num_batches_ = -1;
};

class TablePipeline : public ITablePipeline {
 public:
  explicit TablePipeline(std::shared_ptr<arrow::Table> table) {
    batches_.SetProducerNum(0);
    schema_ = table->schema();
    length_ = table->num_rows();
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    VINEYARD_CHECK_OK(TableToRecordBatches(table, &batches));
    num_batches_ = batches.size();

    for (auto& batch : batches) {
      batches_.Put(batch);
    }
  }

  Status Next(std::shared_ptr<arrow::RecordBatch>& batch) override {
    if (batches_.Size() == 0) {
      return Status::StreamDrained();
    }
    if (batches_.Get(batch)) {
      return Status::OK();
    }
    return Status::StreamDrained();
  }

 private:
  grape::BlockingQueue<std::shared_ptr<arrow::RecordBatch>> batches_;
};

class ConcatTablePipeline : public ITablePipeline {
 public:
  explicit ConcatTablePipeline(
      std::vector<std::shared_ptr<ITablePipeline>> sources,
      const std::shared_ptr<arrow::Schema> schema = nullptr) {
    if (schema == nullptr) {
      schema_ = sources[0]->schema();
    } else {
      schema_ = schema;
    }
    length_ = 0;
    num_batches_ = 0;
    for (auto const& pipe : sources) {
      if (pipe == nullptr) {
        continue;
      }
      sources_.push_back(pipe);
      length_ += pipe->length();
      num_batches_ += pipe->num_batches();
    }
  }

  Status Next(std::shared_ptr<arrow::RecordBatch>& batch) override {
    std::map<std::thread::id, thread_local_item_t>::iterator currentloc;
    {
      // create or find the "current" pointer
      std::lock_guard<std::mutex> lock(current_mutex_);
      currentloc = currents_.find(std::this_thread::get_id());
      if (currentloc == currents_.end()) {
        currents_[std::this_thread::get_id()] = std::make_pair(-1, nullptr);
        currentloc = currents_.find(std::this_thread::get_id());
      }
    }

    // now each thread has its own "current"
    std::pair<int, std::shared_ptr<ITablePipeline>>& current =
        currentloc->second;
    if (current.second == nullptr) {
      if (current.first >= static_cast<int>(sources_.size()) - 1) {
        return Status::StreamDrained();
      }
      current.first += 1;
      current.second = sources_[current.first];
    }
    auto status = current.second->Next(batch);
    if (status.ok() || !status.IsStreamDrained()) {
      // propagate the error
      return status;
    }
    // retry to get a new "current"
    current.second = nullptr;
    return Next(batch);
  }

 private:
  std::vector<std::shared_ptr<ITablePipeline>> sources_;
  // poor man's thread local
  std::mutex current_mutex_;
  using thread_local_item_t =
      std::pair<int /* head */, std::shared_ptr<ITablePipeline>>;
  std::map<std::thread::id, thread_local_item_t> currents_;
};

template <typename S = std::nullptr_t>
class MapTablePipeline : public ITablePipeline {
 public:
  using state_t = S;
  using task_t = std::function<Status(
      const std::shared_ptr<arrow::RecordBatch>& from, std::mutex& mu,
      state_t& state, std::shared_ptr<arrow::RecordBatch>& to)>;

  explicit MapTablePipeline(
      const std::shared_ptr<ITablePipeline> from, task_t task,
      const state_t initial_state = {},
      const std::shared_ptr<arrow::Schema> schema = nullptr)
      : from_(from), task_(task), state_(initial_state) {
    if (schema == nullptr) {
      schema_ = from->schema();
    } else {
      schema_ = schema;
    }
    length_ = from->length();
    num_batches_ = from->num_batches();
  }

  Status Next(std::shared_ptr<arrow::RecordBatch>& batch) override {
    std::shared_ptr<arrow::RecordBatch> from;
    RETURN_ON_ERROR(from_->Next(from));
    RETURN_ON_ERROR(task_(from, state_mutex_, state_, batch));
    return Status::OK();
  }

 private:
  std::shared_ptr<ITablePipeline> from_;
  task_t task_;
  std::mutex state_mutex_;
  state_t state_;
};

class FilterMapPipeline : public ITablePipeline {
 public:
  using task_t =
      std::function<bool(const std::shared_ptr<arrow::RecordBatch>& from)>;

  explicit FilterMapPipeline(
      std::shared_ptr<ITablePipeline> from, task_t task,
      const std::shared_ptr<arrow::Schema> schema = nullptr)
      : from_(from), task_(task) {
    if (schema == nullptr) {
      schema_ = from->schema();
    } else {
      schema_ = schema;
    }
    length_ = from->length();
    num_batches_ = from->num_batches();
  }

  Status Next(std::shared_ptr<arrow::RecordBatch>& batch) override {
    std::shared_ptr<arrow::RecordBatch> from;
    while (true) {
      auto s = from_->Next(from);
      if (s.IsStreamDrained()) {
        return s;
      }
      if (!s.ok()) {
        return s;
      }
      if (task_(from)) {
        batch = from;
        return Status::OK();
      }
    }
    return Status::StreamDrained();
  }

 private:
  std::shared_ptr<ITablePipeline> from_;
  task_t task_;
};

class TablePipelineSink {
 public:
  explicit TablePipelineSink(
      std::shared_ptr<ITablePipeline> from,
      std::shared_ptr<arrow::Schema> schema = nullptr,
      size_t concurrency = std::thread::hardware_concurrency(),
      bool combine_chunks = false)
      : from_(from),
        concurrency_(concurrency),
        combine_chunks_(combine_chunks) {
    if (schema == nullptr) {
      schema_ = from->schema();
    } else {
      schema_ = schema;
    }
  }

  Status Result(std::shared_ptr<arrow::Table>& table) {
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    std::mutex append_mutex;

    auto fn = [&]() -> Status {
      while (true) {
        std::shared_ptr<arrow::RecordBatch> batch;
        auto status = from_->Next(batch);
        if (status.IsStreamDrained()) {
          break;
        }
        RETURN_ON_ERROR(status);
        {
          std::lock_guard<std::mutex> lock(append_mutex);
          batches.push_back(batch);
        }
      }
      return Status::OK();
    };
    ThreadGroup tg(concurrency_);
    for (size_t i = 0; i < concurrency_; ++i) {
      tg.AddTask(fn);
    }
    Status status;
    for (auto const& s : tg.TakeResults()) {
      status += s;
    }
    RETURN_ON_ERROR(status);
    if (combine_chunks_) {
      RETURN_ON_ERROR(CombineRecordBatches(schema_, batches, &table));
    } else {
      RETURN_ON_ERROR(RecordBatchesToTable(schema_, batches, &table));
    }
    return Status::OK();
  }

 private:
  std::shared_ptr<ITablePipeline> from_;
  std::shared_ptr<arrow::Schema> schema_;
  size_t concurrency_;
  bool combine_chunks_;
};

class ShuffleTablePipeline {};

}  // namespace vineyard

#endif  // MODULES_GRAPH_UTILS_TABLE_PIPELINE_H_
