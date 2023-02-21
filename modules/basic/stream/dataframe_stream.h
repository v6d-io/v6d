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

#ifndef MODULES_BASIC_STREAM_DATAFRAME_STREAM_H_
#define MODULES_BASIC_STREAM_DATAFRAME_STREAM_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "basic/ds/dataframe.h"
#include "client/client.h"

namespace vineyard {

class DataframeStream : public BareRegistered<DataframeStream>,
                        public DataFrameStreamBase {
 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<DataframeStream>{new DataframeStream()});
  }

  Status WriteTable(std::shared_ptr<arrow::Table> table);

  Status WriteBatch(std::shared_ptr<arrow::RecordBatch> batch);

  Status WriteDataframe(std::shared_ptr<DataFrame> df);

  Status ReadRecordBatches(
      std::vector<std::shared_ptr<arrow::RecordBatch>>& batches);

  Status ReadTable(std::shared_ptr<arrow::Table>& table);

  Status ReadBatch(std::shared_ptr<arrow::RecordBatch>& batch, const bool copy);

  Status GetHeaderLine(bool& header_row, std::string& header_line);
};

template <>
struct stream_type<DataFrame> {
  using type = DataframeStream;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_STREAM_DATAFRAME_STREAM_H_
